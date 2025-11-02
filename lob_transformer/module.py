import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from lightning.pytorch import LightningModule


class LOBDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int = 100,
                 target_cols: list = ['target'],
                 depth: int = 10):
        super().__init__()
        
        self.data = df.copy()
        self.window_size = window_size
        self.target_cols = target_cols
        self.depth = depth
        
        self._validate_data(self.data)
        
        self.target_data = self._target_normalization(self.data[target_cols])
        target_array = self.target_data[:-min(len(self.target_data), self.window_size)].to_numpy()
        
        # For classification, ensure target is 1D and convert to long tensor
        if len(target_cols) == 1:
            target_array = target_array.flatten()  # Convert to 1D
        self.target_data = self._data_to_tensors(target_array, dtype=torch.long)  # Use long for classification
        
        self.data = self._apply_zscore_normalization(self.data[self.data.columns.difference(target_cols)])
        self.data = self._preprocess_data(self.data)
        self.data = self._data_to_tensors(self.data) # shape: (num_samples, 2, depth*2, window_size)
    
    
    def _validate_data(self, df: pd.DataFrame):
        price_cols = [f'bid_price_{i+1}' for i in range(self.depth)] + [f'ask_price_{i+1}' for i in range(self.depth)]
        size_cols = [f'bid_size_{i+1}' for i in range(self.depth)] + [f'ask_size_{i+1}' for i in range(self.depth)]
        required_columns = self.target_cols + price_cols + size_cols
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
    
    
    def _preprocess_data(self, df: pd.DataFrame) -> np.array:
        snapshots = []
        for i in range(len(df)):
            bids = np.stack([
                df.iloc[i][[f'bid_price_{j+1}' for j in range(self.depth)]].to_numpy(),
                df.iloc[i][[f'bid_size_{j+1}' for j in range(self.depth)]].to_numpy(),
            ], axis=0) # shape: (2, depth)
            asks = np.stack([
                df.iloc[i][[f'ask_price_{j+1}' for j in range(self.depth)]].to_numpy(),
                df.iloc[i][[f'ask_size_{j+1}' for j in range(self.depth)]].to_numpy(),
            ], axis=0) # shape: (2, depth)
            
            snapshot = np.concatenate([bids, asks], axis=1) # shape: (2, depth*2)
            snapshots.append(snapshot)
        
        snapshots = np.stack(snapshots, axis=0) # shape: (num_samples, 2, depth*2)
        
        samples = []
        for i in range(len(snapshots) - self.window_size + 1):
            window = snapshots[i:i + self.window_size] # shape: (window_size, 2, depth*2)
            window = np.transpose(window, (1, 2, 0)) # shape: (2, depth*2, window_size)
            samples.append(window)
        
        return np.stack(samples).astype(np.float32) # shape: (num_samples, 2, depth*2, window_size)
    
    
    def _target_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
    
    
    def _apply_zscore_normalization(self, df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        def rolling_zscore_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            mean = df.rolling(window=window, min_periods=1).mean()
            std = df.rolling(window=window, min_periods=1).std()
            zscore = (df - mean) / std.replace(0, 1)
            zscore = zscore.fillna(0.0)
            return zscore.astype(np.float32)
        
        price_cols = [f'bid_price_{i+1}' for i in range(self.depth)] + [f'ask_price_{i+1}' for i in range(self.depth)]
        size_cols = [f'bid_size_{i+1}' for i in range(self.depth)] + [f'ask_size_{i+1}' for i in range(self.depth)]
        
        df[price_cols] = rolling_zscore_dataframe(df[price_cols])
        df[size_cols] = rolling_zscore_dataframe(df[size_cols])
        
        return df
    
    
    def _data_to_tensors(self, arr: np.array, **kwargs) -> torch.Tensor:
        return torch.tensor(arr, **{
            'dtype': torch.float32,
            **kwargs
        })
    
    
    def to_dataloader(self, num_workers=4, pin_memory=True, **kwargs) -> DataLoader:
        return DataLoader(self, num_workers=num_workers, pin_memory=pin_memory, **kwargs)
    
    
    def __len__(self):
        return max(0, len(self.data) - self.window_size + 1)
    
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.target_data[idx]
        
        return x, y


class StructuredPatchEmbedding(nn.Module):
    def __init__(self, 
                 input_channels: int = 2,
                 patch_size: tuple = (5, 12),
                 embed_dim: int = 128):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = nn.Linear(input_channels * patch_size[0] * patch_size[1], embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 1000, embed_dim))
        
    def forward(self, x):
        batch_size, channels, height, window = x.shape
        patch_height, patch_width = self.patch_size
        
        assert height % patch_height == 0, "Height must be divisible by patch_height"
        assert window % patch_width == 0, "Window must be divisible by patch_width"

        x = (x
             .unfold(2, patch_height, patch_height) # depth dimention
             .unfold(3, patch_width, patch_width)) # window dimention
        x = x.contiguous().view(batch_size, channels, -1, patch_height, patch_width)
        x = x.permute(0, 2, 1, 3, 4).contiguous() # shape: (batch_size, num_patches, channels, patch_height, patch_width)
        x = x.view(batch_size, -1, channels * patch_height * patch_width)
        
        x = self.projection(x)  # shape: (batch_size, num_patches, embed_dim)
        x = x + self.position_embedding[:, :x.size(1)]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        return self.encoder(x)


class LOBTransformer(LightningModule):
    def __init__(self,
                 input_channels: int = 2,
                 patch_size: tuple = (5, 12),
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_transformer_layers: int = 8,
                 lstm_hidden_dim: int = 64,
                 num_classes: int = 3,
                 dropout: float = 0.1,
                 lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = StructuredPatchEmbedding(
            input_channels=input_channels,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        
    def forward(self, x):
        embeddings = self.patch_embedding(x)
        transformer_out = self.transformer(embeddings)
        _, (h_n, c_n) = self.lstm(transformer_out)
        
        return self.classifier(h_n[-1])
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = self.loss_fn(logits, y)

        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def predict_price_movement(self, x):
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            movement_labels = ['down', 'stable', 'up']
            
            results = []
            for i in range(len(predictions)):
                pred_idx = predictions[i].item()
                confidence = probabilities[i][pred_idx].item()
                results.append({
                    'prediction': movement_labels[pred_idx],
                    'confidence': confidence,
                    'probabilities': {
                        'down': probabilities[i][0].item(),
                        'stable': probabilities[i][1].item(),
                        'up': probabilities[i][2].item()
                    }
                })
            
            return results


def calculate_target(df, steps_ahead=12, threshold=0.01/100):
    targets = []
    
    for i in range(len(df)):
        if i + steps_ahead >= len(df):
            targets.append(1)
            continue
        
        current_row = df.iloc[i]
        future_row = df.iloc[i + steps_ahead]
        
        current_bid = current_row.get('bid_price_1', 0)
        current_ask = current_row.get('ask_price_1', 0)
        future_bid = future_row.get('bid_price_1', 0)
        future_ask = future_row.get('ask_price_1', 0)
        
        if current_bid == 0 or current_ask == 0 or future_bid == 0 or future_ask == 0:
            targets.append(1)
            continue
            
        current_mid = (current_bid + current_ask) / 2
        future_mid = (future_bid + future_ask) / 2
        
        price_change = (future_mid - current_mid) / current_mid
        
        if price_change >= threshold:
            targets.append(2)
        elif price_change <= -threshold:
            targets.append(0)
        else:
            targets.append(1)

    return targets


def main():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    supabase_table = os.getenv('SUPABASE_TABLE_FOR_TRAIN_TRANSFORMER')
    
    model_path = 'models'
    max_epochs = 100
    batch_size = 32
    window_size = 60
    
    patch_height = 5
    patch_width = 6
    embed_dim = 128
    num_heads = 8
    num_transformer_layers = 6
    lstm_hidden_dim = 64
    dropout = 0.1
    lr = 1e-4
    
    from supabase import create_client, Client
    from supabase.client import ClientOptions
    supabase: Client = create_client(
        supabase_url,
        supabase_key,
        options=ClientOptions(
            postgrest_client_timeout=604800,
            storage_client_timeout=604800
        )
    )
    
    limit = 10000
    df = pd.concat([pd.DataFrame(
        supabase.table(supabase_table)
        .select('*')
        .limit(limit)
        .offset(limit * o)
        .execute()
        .data
    ) for o in range(6)])
    df = df.reset_index()
    
    df['target'] = calculate_target(df, steps_ahead=12, threshold=0.01/100)
    
    train_cutoff = int(len(df) * 0.8)
    val_cutoff = int(len(df) * 0.9)
    
    train_dataset, val_dataset, test_dataset = (
        LOBDataset(df[lambda x: x.index < train_cutoff], window_size=window_size),
        LOBDataset(df[lambda x: (x.index >= train_cutoff) & (x.index < val_cutoff)], window_size=window_size),
        LOBDataset(df[lambda x: x.index >= val_cutoff], window_size=window_size)
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_dataloader, val_dataloader, test_dataloader = (
        train_dataset.to_dataloader(batch_size=batch_size, shuffle=True),
        val_dataset.to_dataloader(batch_size=batch_size, shuffle=False),
        test_dataset.to_dataloader(batch_size=batch_size, shuffle=False)
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-6,
        patience=5,  # Increased patience
        verbose=True,  # Enable verbose for debugging
        mode='min',
        strict=True,  # Raise error if metric is not found
        check_on_train_epoch_end=False  # Check at validation end
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        filename="lobtransformer-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stopping_callback, checkpoint_callback],
        logger=False
    )
    
    lob_transformer = LOBTransformer(
        input_channels=2,
        patch_size=(patch_height, patch_width),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_transformer_layers=num_transformer_layers,
        lstm_hidden_dim=lstm_hidden_dim,
        num_classes=3,
        dropout=dropout,
        lr=lr
    )
    
    trainer.fit(lob_transformer, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    trainer.test(lob_transformer, test_dataloader)
    
    sample_batch = next(iter(test_dataloader))
    sample_x, sample_y = sample_batch
    predictions = lob_transformer.predict_price_movement(sample_x)
    
    print("\n=== Prediction Results ===")
    for i, pred in enumerate(predictions[:3]):
        print(f"Sample {i+1}:")
        print(f"  Prediction: {pred['prediction']}")
        print(f"  Confidence: {pred['confidence']:.4f}")
        print(f"  Probabilities: {pred['probabilities']}")
        print()


if __name__ == '__main__':
    # Example to use
    main()
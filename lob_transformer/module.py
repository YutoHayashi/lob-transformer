from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchmetrics

from lightning.pytorch import LightningModule, Trainer


@dataclass
class LOBDatasetConfig:
    window_size: int
    horizon: int
    threshold: float
    target_cols: list[str]
    depth: int


class LOBDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 config: LOBDatasetConfig):
        super().__init__()
        
        self.data = df.copy()
        self.config = config
        
        self._validate_data(self.data)
        
        self.target_data = self._target_normalization(self.data[config.target_cols])
        target_array = self.target_data[:-(config.window_size - 1)].to_numpy()
        
        # For classification, ensure target is 1D and convert to long tensor
        if len(config.target_cols) == 1:
            target_array = target_array.flatten()  # Convert to 1D
        self.target_data = self._data_to_tensors(target_array, dtype=torch.long)  # Use long for classification
        
        self.data = self._apply_zscore_normalization(self.data[self.data.columns.difference(config.target_cols)], window=config.window_size)
        self.data = self._preprocess_data(self.data)
        self.data = self._data_to_tensors(self.data) # shape: (num_samples, 2, depth*2, window_size)
    
    def _validate_data(self, df: pd.DataFrame):
        price_cols = [f'bid_price_{i+1}' for i in range(self.config.depth)] + [f'ask_price_{i+1}' for i in range(self.config.depth)]
        size_cols = [f'bid_size_{i+1}' for i in range(self.config.depth)] + [f'ask_size_{i+1}' for i in range(self.config.depth)]
        required_columns = self.config.target_cols + price_cols + size_cols
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
    
    def _preprocess_data(self, df: pd.DataFrame) -> np.array:
        snapshots = []
        for i in range(len(df)):
            bids = np.stack([
                df.iloc[i][[f'bid_price_{j+1}' for j in range(self.config.depth)]].to_numpy(),
                df.iloc[i][[f'bid_size_{j+1}' for j in range(self.config.depth)]].to_numpy(),
            ], axis=0) # shape: (2, depth)
            asks = np.stack([
                df.iloc[i][[f'ask_price_{j+1}' for j in range(self.config.depth)]].to_numpy(),
                df.iloc[i][[f'ask_size_{j+1}' for j in range(self.config.depth)]].to_numpy(),
            ], axis=0) # shape: (2, depth)
            
            snapshot = np.concatenate([bids, asks], axis=1) # shape: (2, depth*2)
            snapshots.append(snapshot)
        
        snapshots = np.stack(snapshots, axis=0) # shape: (num_samples, 2, depth*2)
        
        samples = []
        for i in range(len(snapshots) - self.config.window_size + 1):
            window = snapshots[i:i + self.config.window_size] # shape: (window_size, 2, depth*2)
            window = np.transpose(window, (1, 2, 0)) # shape: (2, depth*2, window_size)
            samples.append(window)
        
        return np.stack(samples).astype(np.float32) # shape: (num_samples, 2, depth*2, window_size)
    
    def _target_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
    
    def _apply_zscore_normalization(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        def rolling_zscore_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            mean = df.rolling(window=window, min_periods=1).mean()
            std = df.rolling(window=window, min_periods=1).std()
            zscore = (df - mean) / std.replace(0, 1)
            zscore = zscore.fillna(0.0)
            return zscore.astype(np.float32)
        
        price_cols = [f'bid_price_{i+1}' for i in range(self.config.depth)] + [f'ask_price_{i+1}' for i in range(self.config.depth)]
        size_cols = [f'bid_size_{i+1}' for i in range(self.config.depth)] + [f'ask_size_{i+1}' for i in range(self.config.depth)]
        
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
        return len(self.data)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.target_data[idx]
        
        return x, y


class StructuredPatchEmbedding(nn.Module):
    def __init__(self, 
                 input_channels: int,
                 patch_size: tuple,
                 embed_dim: int):
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
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float):
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
                 input_channels: int,
                 patch_size: tuple,
                 embed_dim: int,
                 num_heads: int,
                 num_transformer_layers: int,
                 lstm_hidden_dim: int,
                 num_classes: int,
                 dropout: float,
                 lr: float,
                 dataset_config: LOBDatasetConfig = field(default_factory=LOBDatasetConfig)):
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
        
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
    
    @classmethod
    def from_dataset(cls, dataset: LOBDataset, **kwargs):
        return cls(dataset_config=dataset.config, **kwargs)
    
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
        
        self.test_accuracy(pred, y)
        self.test_f1(pred, y)
        self.test_recall(pred, y)
        
        self.log('test_loss', loss)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        
        logits = self(x)
        probabilities = F.softmax(logits, dim=1)

        return probabilities
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


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


def train(train_dataset: LOBDataset,
          val_dataset: LOBDataset,
          batch_size: int,
          max_epochs: int,
          patch_height: int,
          patch_width: int,
          embed_dim: int,
          num_heads: int,
          num_transformer_layers: int,
          lstm_hidden_dim: int,
          input_channels: int,
          dropout: float,
          lr: float,
          num_classes: int,
          model_path: str = 'models',
          ckpt_path: str = None):
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    
    train_dataloader, val_dataloader = (
        train_dataset.to_dataloader(batch_size=batch_size, shuffle=True),
        val_dataset.to_dataloader(batch_size=batch_size, shuffle=False),
    )
    dataset_config = train_dataset.config
    
    print(f"DataLoader sizes - Train: {len(train_dataloader)}, Val: {len(val_dataloader)}")
    
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
        filename=f'lobtransformer-h={dataset_config.horizon}-pt={dataset_config.threshold}-w={dataset_config.window_size}-bs={batch_size}-ph={patch_height}-pw={patch_width}-ed={embed_dim}-heads={num_heads}-tlayers={num_transformer_layers}-lstm={lstm_hidden_dim}-dropout={dropout}-lr={lr}' + "{epoch:02d}-{val_loss:.4f}",
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
    
    lob_transformer = LOBTransformer.from_dataset(
        train_dataset,
        input_channels=input_channels,
        patch_size=(patch_height, patch_width),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_transformer_layers=num_transformer_layers,
        lstm_hidden_dim=lstm_hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        lr=lr
    )
    
    trainer.fit(lob_transformer, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=ckpt_path)
    
    return (
        lob_transformer,
        early_stopping_callback,
        checkpoint_callback
    )


def eval(test_dataset: LOBDataset,
         ckpt_path: str):
    test_dataloader = test_dataset.to_dataloader(batch_size=32, shuffle=False)
    lob_transformer = LOBTransformer.load_from_checkpoint(ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        logger=False
    )
    
    return trainer.test(lob_transformer, dataloaders=test_dataloader)


def main():
    import os
    import json
    
    from dotenv import load_dotenv
    load_dotenv()
    model_path = os.getenv('MODEL_PATH', 'models')
    
    import argparse
    parser = argparse.ArgumentParser(description="LOBTransformer Module")
    
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='Mode to run the script: train or eval')
    parser.add_argument('--preset', type=str, required=False, default='debug', help='Preset configuration to use')
    
    parser.add_argument('--csv_path', type=str, required=False, default=None, help='Path to the CSV data file')
    parser.add_argument('--horizon', type=int, required=False, default=None, help='Prediction horizon in number of events')
    parser.add_argument('--threshold', type=float, required=False, default=None, help='Threshold for target calculation')
    parser.add_argument('--batch_size', type=int, required=False, default=None, help='Batch size for DataLoader')
    parser.add_argument('--window_size', type=int, required=False, default=None, help='Window size for LOBDataset')
    parser.add_argument('--max_epochs', type=int, required=False, default=None, help='Maximum number of training epochs')
    parser.add_argument('--patch_height', type=int, required=False, default=None, help='Patch height for StructuredPatchEmbedding')
    parser.add_argument('--patch_width', type=int, required=False, default=None, help='Patch width for StructuredPatchEmbedding')
    parser.add_argument('--embed_dim', type=int, required=False, default=None, help='Embedding dimension for the model')
    parser.add_argument('--num_heads', type=int, required=False, default=None, help='Number of attention heads')
    parser.add_argument('--num_transformer_layers', type=int, required=False, default=None, help='Number of transformer layers')
    parser.add_argument('--lstm_hidden_dim', type=int, required=False, default=None, help='Hidden dimension for LSTM layer')
    parser.add_argument('--input_channels', type=int, required=False, default=None, help='Number of input channels for the model')
    parser.add_argument('--dropout', type=float, required=False, default=None, help='Dropout rate for the model')
    parser.add_argument('--lr', type=float, required=False, default=None, help='Learning rate for the model')
    
    parser.add_argument('--ckpt_path', type=str, required=False, default=None, help='Path to save/load model checkpoints')
    
    args = parser.parse_args()
    
    with open(os.path.join(os.path.dirname(__file__), 'presets.json'), 'r') as f:
        preset = json.load(f).get(args.preset, {})
    
    args = {k: v for k, v in vars(args).items() if v is not None} | preset
    
    df = pd.read_csv(args.get('csv_path')).reset_index()
    
    target_col = 'target'
    df[target_col] = calculate_target(df, steps_ahead=args.get('horizon'), threshold=args.get('threshold'))
    num_classes = df[target_col].nunique()
    print(f"Target distribution:\n{df[target_col].value_counts(normalize=True)}")
    
    lob_dataset_config = LOBDatasetConfig(
        window_size=args.get('window_size'),
        horizon=args.get('horizon'),
        threshold=args.get('threshold'),
        target_cols=[target_col],
        depth=10
    )
    
    train_cutoff = int(len(df) * 0.8)
    val_cutoff = int(len(df) * 0.9)
    
    if args.get('mode') == 'train':
        train_dataset = LOBDataset(df[lambda x: x.index < train_cutoff], config=lob_dataset_config)
        val_dataset = LOBDataset(df[lambda x: (x.index >= train_cutoff) & (x.index < val_cutoff)], config=lob_dataset_config)
        test_dataset = LOBDataset(df[lambda x: x.index >= val_cutoff], config=lob_dataset_config)
        
        lob_transformer, early_stopping_callback, checkpoint_callback = train(
            train_dataset,
            val_dataset,
            batch_size=args.get('batch_size'),
            max_epochs=args.get('max_epochs'),
            patch_height=args.get('patch_height'),
            patch_width=args.get('patch_width'),
            embed_dim=args.get('embed_dim'),
            num_heads=args.get('num_heads'),
            num_transformer_layers=args.get('num_transformer_layers'),
            lstm_hidden_dim=args.get('lstm_hidden_dim'),
            input_channels=args.get('input_channels'),
            dropout=args.get('dropout'),
            lr=args.get('lr'),
            num_classes=num_classes,
            model_path=model_path,
            ckpt_path=args.get('ckpt_path')
        )
        
        eval(
            test_dataset,
            ckpt_path=checkpoint_callback.best_model_path
        )
    elif args.get('mode') == 'eval':
        test_dataset = LOBDataset(df, config=lob_dataset_config)
        
        eval(
            test_dataset,
            ckpt_path=args.get('ckpt_path')
        )


if __name__ == '__main__':
    # Example to use
    main()
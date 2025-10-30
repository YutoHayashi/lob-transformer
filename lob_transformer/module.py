import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import math

from lightning.pytorch import LightningModule


CHANNELS = 4
DEPTH_SIZE = 10


class LOBDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 window_size: int = 100,
                 normalize: bool = True):
        self.data = data
        self.window_size = window_size
        self.normalize = normalize
        
        if self.normalize:
            self.normalized_data = self._apply_zscore_normalization(data)
        else:
            self.normalized_data = data
    
    def _apply_zscore_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized_df = df.copy()
        
        price_size_columns = []
        for depth_level in range(1, DEPTH_SIZE + 1):
            price_size_columns.extend([
                f'bid_price_{depth_level}',
                f'ask_price_{depth_level}',
                f'bid_size_{depth_level}',
                f'ask_size_{depth_level}'
            ])
        
        existing_columns = [col for col in price_size_columns if col in df.columns]
        
        for col in existing_columns:
            normalized_df[col] = pd.to_numeric(normalized_df[col], errors='coerce')
            
            expanding_mean = normalized_df[col].expanding(min_periods=10).mean()
            expanding_std = normalized_df[col].expanding(min_periods=10).std()
            
            expanding_std = expanding_std.fillna(1.0)
            expanding_std = expanding_std.replace(0.0, 1.0)
            
            normalized_df[col] = (normalized_df[col] - expanding_mean) / expanding_std
            normalized_df[col] = normalized_df[col].fillna(0.0)
        
        return normalized_df
    
    def __len__(self):
        return max(0, len(self.data) - self.window_size + 1)
    
    def __getitem__(self, idx):
        normalized_window_data = self.normalized_data.iloc[idx:idx + self.window_size, :]
        window_data = self.data.iloc[idx:idx + self.window_size, :]
        
        item = np.zeros((CHANNELS, DEPTH_SIZE, self.window_size), dtype=np.float32)
        target = int(window_data['target'].iloc[-1] if pd.notna(window_data['target'].iloc[-1]) else 1)
        
        for t in range(self.window_size):
            if t < len(normalized_window_data):
                row = normalized_window_data.iloc[t]
                
                for depth_level in range(1, DEPTH_SIZE + 1):
                    bid_price_col = f'bid_price_{depth_level}'
                    ask_price_col = f'ask_price_{depth_level}'
                    bid_size_col = f'bid_size_{depth_level}'
                    ask_size_col = f'ask_size_{depth_level}'
                    
                    if bid_price_col in row and pd.notna(row[bid_price_col]):
                        item[0, depth_level-1, t] = float(row[bid_price_col])
                    if ask_price_col in row and pd.notna(row[ask_price_col]):
                        item[1, depth_level-1, t] = float(row[ask_price_col])
                    if bid_size_col in row and pd.notna(row[bid_size_col]):
                        item[2, depth_level-1, t] = float(row[bid_size_col])
                    if ask_size_col in row and pd.notna(row[ask_size_col]):
                        item[3, depth_level-1, t] = float(row[ask_size_col])

        return (
            torch.tensor(item, dtype=torch.float32),
            torch.tensor(target, dtype=torch.long)
        )


class StructuredPatchEmbedding(nn.Module):
    def __init__(self, 
                 input_channels: int = 2,
                 patch_height: int = 5,
                 patch_width: int = 10,
                 embed_dim: int = 256):
        super().__init__()
        self.input_channels = input_channels
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.embed_dim = embed_dim
        
        self.patch_size = input_channels * patch_height * patch_width
        self.projection = nn.Linear(self.patch_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 1000, embed_dim))
        
    def forward(self, x):
        batch_size, channels, depth, window = x.shape
        
        price_data = torch.cat([x[:, 0:1, :, :], x[:, 1:2, :, :]], dim=1)
        size_data = torch.cat([x[:, 2:3, :, :], x[:, 3:4, :, :]], dim=1)
        combined_data = torch.cat([price_data, size_data], dim=1)
        
        patches = []
        num_patches_h = depth // self.patch_height
        num_patches_w = window // self.patch_width
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                h_start, h_end = i * self.patch_height, (i + 1) * self.patch_height
                w_start, w_end = j * self.patch_width, (j + 1) * self.patch_width
                
                patch = combined_data[:, :self.input_channels, h_start:h_end, w_start:w_end]
                patch_flat = patch.flatten(start_dim=1)
                patches.append(patch_flat)
        
        if not patches:
            patch = combined_data[:, :self.input_channels, :, :].flatten(start_dim=1)
            if patch.size(1) < self.patch_size:
                padding = torch.zeros(batch_size, self.patch_size - patch.size(1), device=patch.device, dtype=patch.dtype)
                patch = torch.cat([patch, padding], dim=1)
            elif patch.size(1) > self.patch_size:
                patch = patch[:, :self.patch_size]
            patches = [patch]
        
        patches = torch.stack(patches, dim=1)
        embeddings = self.projection(patches)
        
        num_patches = embeddings.size(1)
        pos_embed = self.position_embedding[:, :num_patches, :]
        embeddings = embeddings + pos_embed
        
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        output = self.out_linear(attention_output)
        
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, ff_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class LOBTransformer(LightningModule):
    def __init__(self,
                 input_channels: int = 2,
                 patch_height: int = 5,
                 patch_width: int = 10,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_transformer_layers: int = 6,
                 ff_dim: int = 1024,
                 lstm_hidden_dim: int = 128,
                 num_classes: int = 3,
                 dropout: float = 0.1,
                 lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.num_classes = num_classes
        
        self.patch_embedding = StructuredPatchEmbedding(
            input_channels=input_channels,
            patch_height=patch_height,
            patch_width=patch_width,
            embed_dim=embed_dim
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_transformer_layers > 1 else 0,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        embeddings = self.patch_embedding(x)
        
        for transformer_block in self.transformer_blocks:
            embeddings = transformer_block(embeddings)
        
        lstm_output, (hidden, cell) = self.lstm(embeddings)
        final_output = lstm_output[:, -1, :]
        logits = self.classifier(final_output)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = self.criterion(logits, y)
        
        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = self.criterion(logits, y)
        
        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = self.criterion(logits, y)
        
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


def calculate_target(df, steps_ahead=24, threshold=0.1/100):
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


if __name__ == '__main__':
    # Example to use
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    from torch.utils.data import DataLoader
    
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    supabase_table = 'bitflyer_fx_btc_jpy_orderbook'
    
    model_path = 'models'
    max_epochs = 100
    batch_size = 64
    window_size = 72
    
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
    
    df['target'] = calculate_target(df, steps_ahead=24, threshold=0.1/100)
    
    train_cutoff = int(len(df) * 0.7)
    val_cutoff = int(len(df) * 0.9)
    
    train_dataset, val_dataset, test_dataset = (
        LOBDataset(df[lambda x: x.index < train_cutoff], window_size=window_size),
        LOBDataset(df[lambda x: (x.index >= train_cutoff) & (x.index < val_cutoff)], window_size=window_size),
        LOBDataset(df[lambda x: x.index >= val_cutoff], window_size=window_size)
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_dataloader, val_dataloader, test_dataloader = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-6,
        patience=3,
        verbose=False,
        mode='min'
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
        patch_height=5,
        patch_width=10,
        embed_dim=256,
        num_heads=8,
        num_transformer_layers=6,
        ff_dim=1024,
        lstm_hidden_dim=128,
        num_classes=3,
        dropout=0.1,
        lr=1e-4
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
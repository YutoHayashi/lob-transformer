import numpy as np
import pandas as pd

from lightning.pytorch import LightningModule

class LobTransformer(LightningModule):
    def __init__(self,
                 lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
    
    def configure_optimizers(self):
        import torch.optim as optim
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):
        pass

if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    supabase_table = 'bitflyer_fx_btc_jpy_orderbook'
    
    from supabase import create_client, Client
    supabase: Client = create_client(supabase_url, supabase_key)
    
    response = (
        supabase.table(supabase_table)
        .select('*')
        .limit(10)
        .execute()
    )
    data = response.data
    count = response.count
    
    df = pd.DataFrame(data)
    print(df, df.shape)
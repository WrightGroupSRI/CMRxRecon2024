from torch.utils.data import DataLoader, Dataset
from cmrxrecon.dl.varnet import VarNetLightning
from cmrxrecon.dl.unet import UnetLightning

import pytorch_lightning as pl
import torch

class DummyDataset(Dataset): 
    def __init__(self): 
        super().__init__()
        
    def __len__(self):
        return 100

    def __getitem__(self, index): 
        return (torch.randn(1, 8, 128, 128, dtype=torch.complex64), torch.randn(1, 8, 128, 128, dtype=torch.complex64))

def main():
    dataset = DummyDataset()
    loader = DataLoader(dataset)
    
    model = VarNetLightning(2, cascades=2, unet_chans=2)
    trainer = pl.Trainer(default_root_dir='/home/kadotab/python/CMRxRecon2024/code/cmrxrecon/dl/models/model_weights/')
    trainer.fit(model=model, train_dataloaders=loader)

if __name__ == '__main__': 
    main()

from torch.utils.data import DataLoader, Dataset
from cmrxrecon.dl.varnet import VarNetLightning
from cmrxrecon.dl.unet import UnetLightning
from cmrxrecon.dl.AllContrastDataModule import AllContrastDataModule
from cmrxrecon.dl.SelfSupervsiedDataModule import SelfSupervisedDataModule
import argparse

import pytorch_lightning as pl
import torch

def main(args):

    #data_module = AllContrastDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    ssl_data_module = SelfSupervisedDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    
    model = VarNetLightning(2, cascades=2, unet_chans=4)
    trainer = pl.Trainer(default_root_dir='/home/kadotab/python/CMRxRecon2024/code/cmrxrecon/dl/models/model_weights/', detect_anomaly=True)
    trainer.fit(model=model, datamodule=ssl_data_module)
    #trainer.test(model=model, datamodule=data_module)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()
    main(args)

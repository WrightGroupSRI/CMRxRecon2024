from cmrxrecon.dl.lowrank_varnet import LowRankLightning
from cmrxrecon.dl.varnet import VarNetLightning
from cmrxrecon.dl.unet import UnetLightning
from cmrxrecon.dl.AllContrastDataModule import AllContrastDataModule
from cmrxrecon.dl.SelfSupervsiedDataModule import SelfSupervisedDataModule
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch

def main(args):
    wandb_logger = WandbLogger(project='cmrxrecon', log_model=True, name=args.run_name, save_dir='cmrxrecon/dl/model_weights/')

    data_module = AllContrastDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    
    if args.model == 'lowrank':
        model = LowRankLightning(cascades=2, unet_chans=32, lr=args.lr)
    elif args.model == 'varnet':
        model = VarNetLightning(2)
    elif args.model == 'unet':
        model = UnetLightning(1)
    else: 
        raise ValueError(f'{args.model} not implemented!')

    trainer = pl.Trainer(
            default_root_dir='cmrxrecon/dl/model_weights/',
            max_epochs=50, 
            limit_train_batches=50, 
            limit_val_batches=2,
            logger=wandb_logger
            )
    trainer.fit(model=model, datamodule=data_module)

    #trainer.test(model=model, datamodule=data_module)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='varnet')

    args = parser.parse_args()
    main(args)

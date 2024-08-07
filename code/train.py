from cmrxrecon.dl.lowrank_varnet import LowRankLightning 
from cmrxrecon.dl.varnet import VarNetLightning
from cmrxrecon.dl.unet import UnetLightning
from cmrxrecon.dl.basis_denoiser_spatial import SpatialDenoiser
from cmrxrecon.dl.basis_denoiser_temporal import TemporalDenoiser
from cmrxrecon.dl.AllContrastDataModule import AllContrastDataModule
from cmrxrecon.dl.BasisDataModule import BasisDataLoader
from cmrxrecon.dl.SelfSupervsiedDataModule import SelfSupervisedDataModule
import argparse
from datetime import timedelta, datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.tuner import Tuner
#from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


def main(args):
    wandb_logger = WandbLogger(project='cmrxrecon', log_model=True, name=args.run_name, save_dir='.')
    
    if args.resubmit:
        filename='checkpoint'
    else:
        now = datetime.now()
        filename= now.strftime('%Y-%m-%d_%H') + '_' + args.model + "_" + '{epoch}-{val/loss:.2f}-{val/ssim:.2f}'

    print(filename)
    
    
    checkpoint_callback = ModelCheckpoint(
            dirpath='/home/kadotab/scratch/cmrxrecon_checkpoints/', 
            filename=filename, 
            train_time_interval=timedelta(minutes=30), 
            save_last=True)

    #checkpoint_callback = ModelCheckpoint(dirpath="cmrxrecon/dl/model_weights/", save_top_k=1, monitor="val/loss")
    data_module = AllContrastDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, file_extension=".h5")
    if args.model == 'lowrank':
        model = LowRankLightning(cascades=5, unet_chans=18, lr=args.lr)
        if args.checkpoint_path: 
            model = LowRankLightning.load_from_checkpoint(args.checkpoint_path, lr=args.lr)
    elif args.model == 'varnet':
        model = VarNetLightning(2, lr=args.lr)
    elif args.model == 'unet':
        model = UnetLightning(1, lr=args.lr, chan=32)
    elif args.model == 'spatial':
        model = SpatialDenoiser(lr=args.lr)
        if args.checkpoint_path: 
            model = SpatialDenoiser.load_from_checkpoint(args.checkpoint_path, lr=args.lr)
    elif args.model == 'temporal':
        model = TemporalDenoiser(lr=args.lr)
        if args.checkpoint_path: 
            model = TemporalDenoiser.load_from_checkpoint(args.checkpoint_path, lr=args.lr)
    else:
        raise ValueError(f'{args.model} not implemented!')
    
    #wandb_logger.experiment.update({'model': args.model})

    profiler = PyTorchProfiler(export_to_chrome=True, filename="prof")
    trainer = pl.Trainer(
            default_root_dir='cmrxrecon/dl/model_weights/',
            max_epochs=args.max_epochs, 
            logger=wandb_logger,
            strategy='ddp',
            limit_train_batches=args.limit_batches,
            limit_val_batches=args.limit_batches,
            limit_test_batches=args.limit_batches,
            callbacks=[checkpoint_callback], 
            detect_anomaly=True
            )

    if trainer.global_rank == 0: 
        wandb_logger.experiment.config.update({'model': args.model})
        wandb_logger.watch(model, log="all")
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='varnet')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--limit_batches', type=float, default=1.0)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--resubmit', action='store_true')

    args = parser.parse_args()
    main(args)

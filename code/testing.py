from cmrxrecon.dl.lowrank_varnet import LowRankLightning
from cmrxrecon.dl.varnet import VarNetLightning
from cmrxrecon.dl.unet import UnetLightning
from cmrxrecon.dl.AllContrastDataModule import AllContrastDataModule
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

def main(args):
    # Initialize WandbLogger
    wandb_logger = WandbLogger(project='cmrxrecon', log_model=True, name=args.run_name, save_dir='cmrxrecon/dl/model_weights/')
    
    # Initialize Data Module
    data_module = AllContrastDataModule(data_dir=args.test_data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Load Model Checkpoint
    if args.model == 'lowrank':
        model = LowRankLightning.load_from_checkpoint(args.checkpoint_path, lr=args.lr)
    elif args.model == 'varnet':
        model = VarNetLightning.load_from_checkpoint(args.checkpoint_path, lr=args.lr)
    elif args.model == 'unet':
        model = UnetLightning.load_from_checkpoint(args.checkpoint_path,input_channels=1, lr=args.lr)
    else:
        raise ValueError(f'{args.model} not implemented!')
    
    # Initialize Trainer
    trainer = pl.Trainer(
            default_root_dir='/home/jaykumar/scratch/cmrxrecon/dl/model_weights/',
            logger=wandb_logger,
            strategy='ddp',
            limit_test_batches=args.limit_batches
            )

    # Perform Testing
    trainer.test(model=model, datamodule=data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_data_dir', type=str, default='/home/jaykumar/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='varnet')
    parser.add_argument('--limit_batches', type=float, default=1.0)
    parser.add_argument('--checkpoint_path', type=str, required=True)

    args = parser.parse_args()
    main(args)

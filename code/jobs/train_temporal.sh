#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=250G
#SBATCH --account=def-mchiew
#SBATCH --time=12:00:00
#SBATCH --output=temporal-%j.out
#SBATCH --mail-user=brenden.kadota@gmail.com
#SBATCH --mail-type=BEGIN,FAIL


#cd to this file and run: sbatch train_lowrank.sh, not sure if this fully trains
#and may need more gpu/time


nvidia-smi

# setup virtual environment
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip 
pip install --no-index -r /home/kadotab/requirements2.txt

export WANDB_INIT_TIMEOUT=400


wandb login 536e03500f10b21a872da7b786ab009c9e9320ac
wandb online

srun python /home/kadotab/python/CMRxRecon2024/code/train.py \
    --num_workers 3 \
    --batch_size 12 \
    --model temporal \
    --lr 1e-3 \

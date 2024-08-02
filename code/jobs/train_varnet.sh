#!/bin/bash
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G
#SBATCH --account=def-mchiew
#SBATCH --time=12:00:00
#SBATCH --output=varnet-%j.out
#SBATCH --mail-user=brenden.kadota@gmail.com
#SBATCH --mail-type=BEGIN,FAIL

nvidia-smi

# setup virtual environment
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip 
pip install --no-index -r ~/requirements2.txt

export WANDB_HTTP_TIMEOUT=300
export WANDB_INIT_TIMEOUT=600

wandb login 536e03500f10b21a872da7b786ab009c9e9320ac
wandb online

srun python /home/kadotab/python/CMRxRecon2024/code/train.py \
    --num_workers 3 \
    --model varnet \
    --lr 1e-3

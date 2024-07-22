#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=20000M 
#SBATCH --account=def-mchiew
#SBATCH --time=7:00:00
#SBATCH --output=loupe-%j.out


# setup virtual environment
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip 
pip install --no-index -r ~/requirements2.txt

srun python /home/kadotab/python/CMRxRecon2024/code/get_sensetivity_maps.py

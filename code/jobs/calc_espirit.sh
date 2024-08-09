#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=180G 
#SBATCH --account=def-mchiew
#SBATCH --time=6:00:00
#SBATCH --output=espirit-%j.out
#SBATCH --array=0-3


# setup virtual environment
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip 
pip install --no-index -r ~/requirements2.txt

myArray=("/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/Aorta/TrainingSet/" 
         "/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/Cine/TrainingSet/"
         "/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/Tagging/TrainingSet//"
         "/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/Mapping/TrainingSet/"
     )

srun python /home/kadotab/python/CMRxRecon2024/code/get_sensetivity_maps.py --path ${myArray[$SLURM_ARRAY_TASK_ID]}

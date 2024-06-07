#/bin/bash
# check the enviroment and get the sumbmitter information,
set -e

cd /output
# DO NOT EDIT THIS LINE
python3 /before_run.py

# You can add more commands here to init your enviromentt
# Run your inference code and output the result to /output

python3 /app/main.py --input_dir /input --predict_dir /output/ZF_TASK1 --weights_dir /app/weights.pt --recon_mode zf --challenge validation --task task1

# DO NOT EDIT THIS LINE
python3 /after_run.py

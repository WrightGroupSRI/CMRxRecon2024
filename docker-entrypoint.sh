#/bin/bash
# check the enviroment and get the sumbmitter information,
set -e

cd /output
# DO NOT EDIT THIS LINE
python3 /before_run.py

# You can add more commands here to init your enviromentt
# Run your inference code and output the result to /output

python3 /app/main.py --input_dir /input --predict_dir /output/E2ELOWRANK_TASK2 --weights_dir /app/weights/end_to_end.ckpt --recon_mode ours --challenge validation --task task2
# python3 /app/main.py --input_dir /input --predict_dir /output/UNET_TASK2 --weights_dir /app/weights/epoch=28-step=50054.ckpt --recon_mode unet --challenge validation --task task2
# python3 /app/main.py --input_dir /input --predict_dir /output/CS_TASK2 --weights_dir /app/weights.pt --recon_mode cs --challenge validation --task task2
# python3 /app/main.py --input_dir /input --predict_dir /output/SENSE_TASK2 --weights_dir /app/weights.pt --recon_mode pi --challenge validation --task task2
# python3 /app/main.py --input_dir /input --predict_dir /output/ZF_TASK2 --weights_dir /app/weights.pt --recon_mode zf --challenge validation --task task2

# DO NOT EDIT THIS LINE
python3 /after_run.py

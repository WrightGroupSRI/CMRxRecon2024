import h5py
import os
import numpy as np
import multiprocessing

path = '/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/'

def process_file(file_name):
    print(f'starting {file_name}')
    name, ext = os.path.splitext(file_name)

    if ext == '.mat':
        kspace = None
        keys = None
        with h5py.File(file_name, 'r') as fr:
            keys = list(fr.keys())
            kspace = fr[keys[0]][:]
        
        if kspace.ndim == 5:
            kspace = np.transpose(kspace, (1, 0, 2, 3, 4))

        new_name = os.path.join(os.path.dirname(file_name), name + '.h5')
        with h5py.File(new_name, 'w') as fr:
            fr.create_dataset(keys[0], data=kspace)

        print(f'saved {new_name}')

def get_all_files(path):
    file_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file_name = os.path.join(root, file)
            file_paths.append(file_name)
    return file_paths

if __name__ == "__main__":
    all_files = get_all_files(path)

    # Create a pool of worker processes
    with multiprocessing.Pool(10) as pool:
        pool.map(process_file, all_files)

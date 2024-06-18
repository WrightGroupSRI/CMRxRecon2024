import h5py
import os
import numpy as np
import multiprocessing

path = '/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/'

def process_file(file_name):
    print(f'starting {file_name}')
    try: 
        name, ext = os.path.splitext(file_name)
        if 'mask' in name: 
            return None

        if ext == '.h5':
            kspace = None
            keys = None
            with h5py.File(file_name, 'r') as fr:
                keys = list(fr.keys())
                kspace = fr[keys[0]][:]
            
                shape =  np.array(kspace.shape[1:])
                if len(shape) == 3: 
                    shape = shape[np.newaxis, :, :, :]
                return shape
        else: 
            return None
    except:
        raise  Exception(f'FOUND ERROR {file_name}')


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
        results = pool.map(process_file, all_files)
    
    #results = []
    #for i, file in enumerate(all_files):
    #    results.append(process_file(file))
    #    if i > 500:
    #        break

    results = [result for result in results if result is not None]

    shapes = np.stack(results, axis=0)
    unique_shapes = np.unique(shapes, axis=0)
    print(unique_shapes)



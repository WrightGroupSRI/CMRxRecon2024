import pytest
import torch
import tempfile
from cmrxrecon.dl.dataloaders.AllContrastDataset import AllContrastDataset

def test_file_list_length():
    dataset = AllContrastDataset(
            '/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/'
            )

    assert len(dataset.file_list) == len(dataset)

def test_all_contrasts_loaded():
    dataset = AllContrastDataset(
            '/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/'
            )

    assert len(dataset.datasets) == 8
    
    # all datasets should be populated
    dataset_lengths = [len(volume_dataset) for volume_dataset in dataset.datasets]
    assert not any([length == 0 for length in dataset_lengths])
    

if __name__ == "__main__":
    pytest.main()


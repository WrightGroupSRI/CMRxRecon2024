import pytest
import torch
from cmrxrecon.dl.AllContrastDataModule import pad_to_max_time, ZeroPadKSpace
from cmrxrecon.dl.dataloaders.VolumeDataset import VolumeDataset


def test_zero_padding(): 
    path = '/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/Aorta/'
    dataset = VolumeDataset(path, False, True, '4', 'aorta_sag')
    data = dataset[0]
    padded_data = ZeroPadKSpace()(data)
    
    # ensure all the shapes are padded correctly
    assert padded_data[0].shape == (256, 512)
    assert padded_data[1].shape == (256, 512)
    assert padded_data[2].shape == (256, 512)


    # Calculate padding sizes
    pad_top = (256 - data[0].shape[-2]) // 2
    pad_bottom = 256 - data[0].shape[-2] - pad_top
    pad_left = (512 - data[0].shape[-1]) // 2
    pad_right = 512 - data[0].shape[-1] - pad_left

    # ensure padded region is unchanged
    torch.testing.assert_close(padded_data[0][..., pad_top:pad_top + data[0].shape[-2], pad_left:pad_left + data[0].shape[-1]], data[0])
    torch.testing.assert_close(padded_data[1][..., pad_top:pad_top + data[1].shape[-2], pad_left:pad_left + data[1].shape[-1]], data[1])

    # ensure padding is zero
    # Check the padded regions are zero
    assert torch.all(padded_data[0][...,:pad_top, :] == 0)
    assert torch.all(padded_data[0][...,pad_top+data[0].shape[-2]:, :] == 0)
    assert torch.all(padded_data[0][...,:, :pad_left] == 0)
    assert torch.all(padded_data[0][...,:, pad_left+data[0].shape[-1]:] == 0)

    torch.testing.assert_close(data[2][data[2] != 0], torch.ones_like(data[2][data[2] != 0]))


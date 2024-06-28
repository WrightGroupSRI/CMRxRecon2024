import torch 
from torch.utils.data import Dataset
import numpy as np
from typing import Callable, Optional

class SelfSupervisedDatset(Dataset):
    """
    Self Supervised Datsert

    Self supervised datset decorator. Takes a dataset as input and converts it to
    a self-supervised representation.
    """

    def __init__(self, dataset: Dataset, transforms:Optional[Callable] = None) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index): 
        fully_sampled, undersampled = self.dataset[index]
        pdf = gen_pdf_bern(fully_sampled.shape[-1], fully_sampled.shape[-2], 1/2, 8, 10)
        pdf = pdf[None, :, :]
        mask = get_mask_from_distribution(pdf)
        mask = mask[None, :, :, :]
        doub_under = undersampled * torch.from_numpy(mask)
        loss_set = doub_under * (~mask)

        return doub_under, loss_set

         
        
def gen_pdf_bern(nx, ny, delta, p, c_sq):
    # generates 2D polynomial variable density with sampling factor delta, fully sampled central square c_sq
    xv, yv = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    r = np.sqrt(xv ** 2 + yv ** 2)
    r /= (np.max(r) + 2/ny)

    prob_map = (1 - r) ** p
    prob_map[prob_map > 1] = 1
    prob_map[prob_map < 0] = 0

    prob_map = scale_pdf(prob_map, 1/delta, c_sq)

    assert prob_map.max() <= 1 and prob_map.min() >= 0
    assert np.isclose(prob_map.mean(), delta, 1e-2, 0), f'got {prob_map.mean()}'

    return prob_map


def scale_pdf(input_prob, R, center_square):
    input_prob[input_prob > 1] = 1
    prob_map = input_prob.copy() 
    nx, ny = prob_map.shape[-2:]


    if prob_map.ndim == 2: 
        prob_map = np.expand_dims(prob_map, axis=0)

    prob_one_index = prob_map == 1

    probability_sum = prob_map.sum((-1, -2)) - np.sum(prob_map[prob_one_index]) 

    probability_total = nx * ny * (1/R) - np.sum(prob_map[prob_one_index])

    for i in range(probability_sum.size): 
        if probability_sum[i] > probability_total: 
            scaling_factor = probability_total/probability_sum[i]
            scaled_prob = prob_map[i, ...] * scaling_factor
        else:
            inverse_total = nx * ny *(1 - (1/R))
            inverse_sum = nx * ny - probability_sum[i] + np.sum(prob_map[prob_one_index])
            scaling_factor = inverse_total / inverse_sum
            scaled_prob = 1 - (1 - prob_map[i, ...]) * scaling_factor
        
        # sometimes prob scaling can cause bad values, check and fix
        scaled_prob[prob_one_index[i, :, :]] = 1
        if np.any(scaled_prob > 1):
            scaled_prob = scale_pdf(scaled_prob, R, center_square)

        prob_map[i, ...] = scaled_prob

    if input_prob.ndim == 2:
        prob_map = np.squeeze(prob_map)


    return prob_map


def get_mask_from_distribution(prob_map):
    rng = np.random.default_rng()
    mask = rng.binomial(1, prob_map).astype(bool)
    return mask

from cmrxrecon.dl.AllContrastDataModule import AllContrastDataModule, AllContrastDataset, NormalizeKSpace, ZeroPadKSpace
from torchvision.transforms import Compose
from cmrxrecon.utils import ifft_2d_img
import torch
from cmrxrecon.dl.basis_denoiser_spatial import cg_data_consistency_R
from torch.utils.data import random_split

class BasisDataLoader(AllContrastDataModule): 
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 1, num_workers: int = 0, is_spatial=True):
        super().__init__(data_dir, batch_size, num_workers)
        self.is_spatial = is_spatial

    def setup(self, stage): 
        all_contrast_full = AllContrastDataset(
                self.data_dir, 
                train=True,
                transforms=Compose([NormalizeKSpace(), ZeroPadKSpace(), GetSingularVectors(self.is_spatial)]),
                task_one=False
                )

        self.all_contrast_train, self.all_contrast_val, self.all_contrast_test = random_split(
            all_contrast_full, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
        )

class GetSingularVectors(object):
    def __init__(self, is_spatial): 
        self.is_spatial = is_spatial

    def __call__(self, sample:torch.Tensor):
        # dimensions [t, h, w]
        under, fully_sampled, sense = sample
        
        temporal, spatial = estimate_inital_bases(under, sense, under != 0)

        fully_sampled_images = (ifft_2d_img(fully_sampled)* sense.conj()).sum(1) / (sense.conj() * sense).sum(1)
        gt_temporal, gt_spatial_basis = get_singular_vectors(fully_sampled_images)
        if self.is_spatial:
            input = spatial
            output = gt_spatial_basis
        else: 
            input = temporal
            output = gt_temporal

        return input, output, sense


def get_singular_vectors(data):
    t, h, w = data.shape

    temporal_basis, _, spatial_basis = torch.linalg.svd(data.view(b, t, h*w), full_matrices=False, driver='gesvdj')
    components = 3 #(singular_values > singular_values[0]*self.singular_cuttoff).numel()
    temporal_basis = temporal_basis[:, :, :components]
    spatial_basis = spatial_basis[:, :components, :]
    spatial_basis = spatial_basis.view(components, h, w)
    
    return temporal_basis, spatial_basis

def estimate_inital_bases(reference_k, sense_maps, mask):
    masked_k = get_center_masked_k_space(reference_k) 
    masked_k = (ifft_2d_img(masked_k) * sense_maps.conj()).sum(1) / (sense_maps * sense_maps.conj()).sum(1)
    temporal_basis, spatial_basis = get_singular_vectors(masked_k)
    cg_spatial = cg_data_consistency_R(iterations=4, lambda_reg=1).to(spatial_basis.device)

    spatial_basis = cg_spatial(reference_k, torch.zeros_like(spatial_basis, requires_grad=False, device=spatial_basis.device), sense_maps, temporal_basis, mask)
    return spatial_basis, temporal_basis

def get_center_masked_k_space(k_space):
    center_mask = torch.zeros_like(k_space, dtype=torch.bool)
    center_y, center_x = center_mask.shape[-2:]
    center_mask[:, :, center_y//2-8:center_y//2+8, center_x//2-8:center_x//2+8] = 1
    masked_k = k_space * center_mask
    return masked_k


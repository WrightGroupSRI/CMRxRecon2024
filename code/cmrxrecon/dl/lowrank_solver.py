import torch.nn as nn
import torch
from typing import Tuple
from functools import partial
import wandb

from cmrxrecon.dl.unet import Unet
from cmrxrecon.dl.resnet import ResNet
from cmrxrecon.metrics import metrics
from torch.fft import ifftshift, fftshift, fft2, ifft2
import pytorch_lightning as pl 
from torchvision.utils import make_grid



class LowRankSolver(nn.Module):
    def __init__(self, 
                 spatial_denoiser: torch.nn.Module, 
                 temporal_denoiser: torch.nn.Module,
                 cascades:int = 5
                 ):
        super().__init__()

        # module for cascades
        self.cascade = nn.ModuleList()
        self.lambda_reg = 1e-1
        
        # populate cascade with model backbone
        spatial_denoiser = spatial_denoiser
        #temporal_denoiser = partial(ResNet, 2, 2, chans=unet_chans//2, dimension='1d')
        temporal_denoiser = temporal_denoiser
        self.cascades = nn.ModuleList(
            [model_step(spatial_denoiser, temporal_denoiser, self.lambda_reg) for _ in range(cascades)]
        )

        # model to estimate sensetivities

    

    def get_singular_vectors(self, data):
        b, t, h, w = data.shape

        temporal_basis, _, spatial_basis = torch.svd(data.view(b, t, h*w))
        spatial_basis = spatial_basis.conj().transpose(-1, -2)
        components = self.singular_cuttoff #(singular_values > singular_values[0]*self.singular_cuttoff).numel()
        temporal_basis = temporal_basis[:, :, :components]
        spatial_basis = spatial_basis[:, :components, :]
        spatial_basis = spatial_basis.view(b, components, h, w)
        
        return temporal_basis, spatial_basis
    
    def get_center_masked_k_space(self, k_space):
        center_mask = torch.zeros_like(k_space, dtype=torch.bool)
        center_y, center_x = center_mask.shape[-2:]
        center_mask[:, :, :, center_y//2-8:center_y//2+8, center_x//2-8:center_x//2+8] = 1
        masked_k = k_space * center_mask
        return masked_k

    def combined_bases(self, spatial_basis, temporal_basis):
        b, components, h, w = spatial_basis.shape
        b, t, components = temporal_basis.shape
        images = (temporal_basis @ spatial_basis.reshape(b, components, h*w)).view(b, t, h, w)
        return images
        

    def estimate_inital_bases(self, reference_k, sense_maps, mask):
        masked_k = self.get_center_masked_k_space(reference_k) 
        masked_k = (ifft_2d_img(masked_k) * sense_maps.conj()).sum(2) / (sense_maps * sense_maps.conj()).sum(2)
        temporal_basis, spatial_basis = self.get_singular_vectors(masked_k)
        cg_spatial = cg_data_consistency_R(iterations=10, lambda_reg=1e-1).to(spatial_basis.device)

        spatial_basis = cg_spatial(reference_k, torch.zeros_like(spatial_basis, requires_grad=False, device=spatial_basis.device), sense_maps, temporal_basis, mask)
        return spatial_basis, temporal_basis

    # k-space sent in [B, T, C, H, W]
    def forward(self, reference_k, mask, sense_maps):
        # get sensetivity maps
        assert not torch.isnan(reference_k).any()
        assert not torch.isnan(mask).any()
        assert not torch.isnan(sense_maps).any()

        spatial_basis, temporal_basis = self.estimate_inital_bases(reference_k, sense_maps, mask)

        for i, cascade in enumerate(self.cascades):
            # go through ith model cascade
            new_spatial_basis, new_temporal_basis = cascade(reference_k, spatial_basis, temporal_basis, sense_maps, mask)
            spatial_basis = new_spatial_basis
            temporal_basis = new_temporal_basis
        
        images = self.combined_bases(spatial_basis, temporal_basis)
        coil_images = images[:, :, None, :, :] * sense_maps
        estimated_k_space = fft_2d_img(coil_images)
        
        #only estimate k-space locations that are unsampled
        return estimated_k_space


"""
Parent class for conjugate gradient data consistency steps

This class should be inherited to solve for the L and R subproblems using 
conjugate gradient

Examples:
    # Description of my example.
    use_it_this_way(arg1, arg2)
"""
class cg_data_consistency(nn.Module):
    def __init__(self, iterations:int = 30, error_tolerance: float = 0.001, lambda_reg=0.0) -> None:
        super().__init__()
        self.iterations = iterations
        self.lambda_reg = nn.Parameter(torch.Tensor([lambda_reg]), requires_grad=True)
        self.error_tolerance = error_tolerance

    def solve(self, system_matrix, b, sensetivity, basis, mask) -> torch.Tensor:
        # start guess at zero
        z = 0
        # Residual is equal to b? Make sense if our strating guess is 0
        r = b
        # search direction is residual from CG
        p = r
        # This is some terms that need to be calculated each iteration
        # calculates r^T r 
        rs_old = torch.dot(r.conj().view(-1), r.view(-1))

        for _ in range(self.iterations):
            Ap = system_matrix(p, sensetivity, basis, mask)
            step_size = rs_old/(torch.dot(p.conj().view(-1), Ap.view(-1)))
            z = z + step_size*p
            r = r - step_size*Ap
            rsnew = torch.dot(torch.conj(r.view(-1)), r.view(-1,))
            
            p = r + (rsnew/(rs_old))*p
            rs_old = rsnew
            if rs_old.abs().max() < 1e-6:
                break

        return z
        
    def pass_through_forward_model(self, data, sensetivities, mask):
        coil_images = sensetivities * data.unsqueeze(2)
        k_space = fft_2d_img(coil_images)
        masked_k = mask * k_space
        return masked_k
    
    def pass_through_adjoint_forward_model(self, data, sensetivities):
        images = ifft_2d_img(data)
        coil_combined = torch.sum(torch.conj(sensetivities)*images, dim=2)
        return coil_combined

"""
Implementation of cg for the spatial basis

Solve the spatial basis problem ||Y - ALR|| + lambda||R - model(R)|| where we solve for R here 

"""
class cg_data_consistency_R(cg_data_consistency):
    def __init__(self, iterations:int = 30, error_tolerance: float = 0.001, lambda_reg=0.0) -> None:
        super().__init__(iterations=iterations, error_tolerance=error_tolerance, lambda_reg=lambda_reg)

    def forward(self, initial_data, model_output, sensetivity, time_basis, mask):
        b = time_basis.permute(0, 2, 1).conj() @ self.pass_through_adjoint_forward_model(initial_data, sensetivity).view(initial_data.shape[0], initial_data.shape[1], -1)
        b = b.view(b.shape[0], b.shape[1], initial_data.shape[-2], initial_data.shape[-1])
        b += self.lambda_reg*model_output

        output_spatial_basis = super().solve(self.system_matrix_for_R, b, sensetivity, time_basis, mask)
        return output_spatial_basis

    def system_matrix_for_R(self, data, sensetivity, L, mask): 
        b, sv, h, w = data.shape
        expanded_data = L @ data.view(b, sv, h*w)
        expanded_data = expanded_data.reshape(b, L.shape[1], h, w)

        masked_k = self.pass_through_forward_model(expanded_data, sensetivity, mask)
        combined_img = self.pass_through_adjoint_forward_model(masked_k, sensetivity)

        final_basis = torch.transpose(torch.conj(L), -1, -2) @ combined_img.reshape(b, L.shape[1], h*w)
        final_basis = final_basis.view(b, sv, h, w)

        final_basis += self.lambda_reg * data
        return final_basis

"""
Solves for the subproblem for L

Solves the problem ||Y - ALR|| - ||L - model(L)|| where we solve for L
"""

class cg_data_consistency_L(cg_data_consistency):
    def __init__(self, iterations:int = 30, error_tolerance: float = 0.001, lambda_reg=0.0) -> None:
        super().__init__(iterations, error_tolerance, lambda_reg)

    def forward(self, initial_data, model_output, sensetivity, spatial_basis, mask):
        Ah_b = self.pass_through_adjoint_forward_model(initial_data, sensetivity).view(initial_data.shape[0], initial_data.shape[1], -1)
        b = Ah_b @ spatial_basis.view(spatial_basis.shape[0], spatial_basis.shape[1], -1).permute(0, 2, 1).conj()
        b = b.view(b.shape[0], b.shape[1], spatial_basis.shape[1])
        b += self.lambda_reg*model_output
        output_time_basis = super().solve(self.system_matrix_for_L, b, sensetivity, spatial_basis, mask)
        return output_time_basis

    def system_matrix_for_L(self, data, sensetivity, R, mask): 
        b, t, sv = data.shape
        b, sv, h, w = R.shape
        R_matrix = R.view(b, sv, h*w)
        expanded = data @ R_matrix
        expanded = expanded.view(b, t, h, w)
        masked_k = self.pass_through_forward_model(expanded, sensetivity, mask)
        combined_img = self.pass_through_adjoint_forward_model(masked_k, sensetivity)

        final_basis = combined_img.view(b, t, h*w) @ torch.conj(R_matrix.transpose(-1, -2))
        final_basis.view(b, sv, t)

        final_basis += self.lambda_reg * data
        return final_basis


class model_step(nn.Module):
    def __init__(self, spatial_model: nn.Module, temporal_model: nn.Module, lambda_reg) -> None:
        super().__init__()
        self.spatial_model = spatial_model
        self.temporal_model = temporal_model
        self.cg_spatial = cg_data_consistency_R(iterations=10, lambda_reg=lambda_reg)
        self.cg_temporal = cg_data_consistency_L(iterations=10, lambda_reg=lambda_reg)

    def pass_spatial_basis_through_model(self, spatial_basis):
        b, sv, h, w = spatial_basis.shape
        # norm spatial basis
        normed_basis, mean, std = self.norm(spatial_basis)
        
        normed_basis = normed_basis.view(b * sv, 1, h, w)
        
        # view as real from complex data and pass through model
        normed_basis = view_as_real(normed_basis)
        denoised_spatail_basis = self.spatial_model(normed_basis) 
        denoised_spatail_basis = view_as_complex(denoised_spatail_basis)
        denoised_spatail_basis = denoised_spatail_basis.view(b, sv, h, w)
        
        # solve conjugate graident problem
        denoised_spatail_basis = self.unnorm(denoised_spatail_basis, mean, std)
        return denoised_spatail_basis

    def pass_temporal_basis_through_model(self, temporal_basis):
        ## reshape temporal basis for passing through network
        temporal_basis = temporal_basis.unsqueeze(1).permute(0, 1, 3, 2)
        temporal_basis, mean, std = self.norm(temporal_basis, dims=1)

        temporal_basis = view_as_real(temporal_basis)
        b, cmplx, sv, t = temporal_basis.shape
        temporal_basis = temporal_basis.view(b*sv, cmplx, t)
        #
        ## pass through network
        denoised_temporal_basis = self.temporal_model(temporal_basis) 
        denoised_temporal_basis = denoised_temporal_basis.view(b, cmplx, sv, t)

        denoised_temporal_basis = view_as_complex(denoised_temporal_basis)

        denoised_temporal_basis = self.unnorm(denoised_temporal_basis, mean, std)
        denoised_temporal_basis = denoised_temporal_basis.permute(0, 1, 3, 2).squeeze(1)
        return denoised_temporal_basis

    # sensetivities data [B, contrast, C, H, W]
    def forward(self, Y, spatial_basis, temporal_basis, sensetivities, mask):
        denoised_spatial_basis = spatial_basis + self.spatial_model(spatial_basis)
        spatial_basis = self.cg_spatial(Y, denoised_spatial_basis, sensetivities, temporal_basis, mask)
        
        denoised_temporal_basis = temporal_basis + self.temporal_model(temporal_basis)
        ## solve temporal problem with CG
        temporal_basis = self.cg_temporal(Y, denoised_temporal_basis, sensetivities, spatial_basis, mask)

        return spatial_basis, temporal_basis

    
    # is this not just instance norm?
    def norm(self, x: torch.Tensor, dims=2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # instance norm
        if dims == 2:
            mean = x.mean(dim=(-1, -2), keepdim=True)
            std = x.std(dim=(-1, -2), keepdim=True) + 1e-9
        elif dims == 1: 
            mean = x.mean(dim=(-1), keepdim=True)
            std = x.std(dim=(-1), keepdim=True) + 1e-9
        else: 
            raise ValueError(f"Got invalid argumetn for dims {dims}")

        x = (x - mean) / std
        return x, mean, std


    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        x = x * std + mean
        return x


###############################################################################
############# HELPER FUNCTIONS ################################################
###############################################################################

fft_2d_img = lambda x, axes=[-1, -2]: fftshift(ifft2(ifftshift(x, dim=axes), dim=axes, norm='ortho'), dim=axes) 
ifft_2d_img = lambda x, axes=[-1, -2]: ifftshift(fft2(fftshift(x, dim=axes), dim=axes, norm='ortho'), dim=axes) 

def view_as_real(data): 
    shape = data.shape
    real_data = torch.view_as_real(data)
    real_data = real_data.reshape(shape[0], shape[1]*2, *shape[2:])
    return real_data

def view_as_complex(data):
    shape = data.shape
    data = data.view(shape[0], shape[1]//2, *shape[2:], 2)
    complex_data = torch.view_as_complex(data)
    return complex_data

    


import torch.nn as nn
import torch
from typing import Tuple
from functools import partial
import wandb

from cmrxrecon.dl.fastmri_unet import Unet
from cmrxrecon.dl.resnet import ResNet
from cmrxrecon.metrics import metrics
from torch.fft import ifftshift, fftshift, fft2, ifft2
import pytorch_lightning as pl 
from torchvision.utils import make_grid



class SpatialDenoiser(pl.LightningModule):
    def __init__(self, lr=1e-3, single_channel=False):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.single_channel = single_channel
        if single_channel:
            self.model = Unet(2, 2, chans=64)
        else:
            self.model = Unet(6, 6, chans=64)
        self.loss_fn = lambda x, y: torch.nn.functional.mse_loss((x), (y))

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_index: int): 
        undersampled, fully_sampled, sense = batch
        
        spatial_basis, _ = self.estimate_inital_bases(undersampled, sense, undersampled != 0)
        #spatial_basis, mean, std = self.norm(spatial_basis)
        spatial_basis = spatial_basis / spatial_basis.abs().amax((-1, -2), keepdim=True)

        spatial_basis = view_as_real(spatial_basis)

        b, sv, h, w = spatial_basis.shape
        if self.single_channel: 
            spatial_basis = spatial_basis.reshape(b*sv//2, 2, h, w)

        output = self.model(spatial_basis)
        denoised_spatial = spatial_basis + output

        if self.single_channel: 
            denoised_spatial = denoised_spatial.reshape(b, sv, h, w)
            spatial_basis = spatial_basis.reshape(b, sv, h, w)
        
        fully_sampled_images = (ifft_2d_img(fully_sampled)* sense.conj()).sum(2) / (sense.conj() * sense + 1e-6).sum(2)
        fully_sampled_images[torch.isnan(fully_sampled_images)] = 0
        _, gt_spatial_basis = self.get_singular_vectors(fully_sampled_images)
        gt_spatial_basis = gt_spatial_basis / gt_spatial_basis.abs().amax((-1, -2), keepdim=True)
        #gt_spatial_basis, _, _ = self.norm(gt_spatial_basis)
        gt_spatial_basis = view_as_real(gt_spatial_basis.resolve_conj())

        ssim = metrics.calculate_ssim(denoised_spatial, gt_spatial_basis, self.device)
        ssim_loss = (1  - ssim ) 
        l1_loss = self.loss_fn(denoised_spatial, gt_spatial_basis)
        loss = ssim_loss + l1_loss

        self.log('train/ssim_loss', ssim_loss, on_step=True, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        self.log('train/l1_loss', l1_loss, on_step=True, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        self.log('train/loss', loss, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        
        if batch_index == 0:  # Log only for the first batch in each epoch
            with torch.no_grad():
                denoised_spatial = view_as_complex(denoised_spatial)
                gt_spatial_basis = view_as_complex(gt_spatial_basis)
                spatial_basis = view_as_complex(spatial_basis)
                
                #gt_spatial_basis = self.unnorm(gt_spatial_basis, mean, std)
                #denoised_spatial = self.unnorm(denoised_spatial, mean, std)
                #spatial_basis = self.unnorm(spatial_basis, mean, std)

                grid = self.prepare_images(denoised_spatial.abs(), gt_spatial_basis.abs().max()/2)
                self.logger.log_image("train/estimate_bases", [wandb.Image(grid, caption="Validation Ground Truth Images")])
                # imgs [b, t, h, w]
                grid = self.prepare_images(gt_spatial_basis.abs(), gt_spatial_basis.abs().max()/2)
                self.logger.log_image("train/gt_spatial_bases", [wandb.Image(grid, caption="Estimated Images")])

                grid = self.prepare_images(spatial_basis.abs(), gt_spatial_basis.abs().max()/2)
                self.logger.log_image("train/zero_filled", [wandb.Image(grid, caption="Zero filled images")])
                
                # [b, t, s, h, w]
                plot_sense = sense[0, 0, :, :, :].unsqueeze(1)

                grid = make_grid(plot_sense.abs())
                self.logger.log_image("train/sense_maps", [wandb.Image(grid, caption="sense")])

                grid = make_grid(plot_sense.angle())
                self.logger.log_image("train/sense_maps_angle", [wandb.Image(grid, caption="sense")])

        return loss


    def validation_step(self, batch, batch_index): 
        undersampled, fully_sampled, sense = batch
        
        spatial_basis, _ = self.estimate_inital_bases(undersampled, sense, undersampled != 0)
        #spatial_basis, mean, std = self.norm(spatial_basis)
        spatial_basis = spatial_basis / spatial_basis.abs().amax((-1, -2), keepdim=True)
        spatial_basis = view_as_real(spatial_basis)

        output = self.model(spatial_basis)
        denoised_spatial = spatial_basis + output
        
        fully_sampled_image = (ifft_2d_img(fully_sampled)* sense.conj()).sum(2) / (sense.conj() * sense + 1e-6).sum(2)
        _, gt_spatial_basis = self.get_singular_vectors(fully_sampled_image)
        #gt_spatial_basis, _, _ = self.norm(gt_spatial_basis)
        gt_spatial_basis = gt_spatial_basis / gt_spatial_basis.abs().amax((-1, -2), keepdim=True)
        gt_spatial_basis = view_as_real(gt_spatial_basis.resolve_conj())

        ssim_loss = metrics.calculate_ssim(denoised_spatial, gt_spatial_basis, self.device)
        l1_loss = self.loss_fn(denoised_spatial, gt_spatial_basis)
        loss = l1_loss + (1 - ssim_loss) 

        self.log('val/ssim', ssim_loss, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        self.log('val/l1_loss', l1_loss, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        self.log('val/loss', loss, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)

        loss = ssim_loss + loss

        if batch_index == 0:  # Log only for the first batch in each epoch
            with torch.no_grad():
                denoised_spatial = view_as_complex(denoised_spatial)
                gt_spatial_basis = view_as_complex(gt_spatial_basis)
                spatial_basis = view_as_complex(spatial_basis)

                #gt_spatial_basis = self.unnorm(gt_spatial_basis, mean, std)
                #denoised_spatial = self.unnorm(denoised_spatial, mean, std)
                #spatial_basis = self.unnorm(spatial_basis, mean, std)
                
                grid = self.prepare_images(denoised_spatial.abs(), gt_spatial_basis.abs().max()/2)
                self.logger.log_image("val/estimate_bases", [wandb.Image(grid, caption="Validation Ground Truth Images")])
                # imgs [b, t, h, w]
                grid = self.prepare_images(gt_spatial_basis.abs(), gt_spatial_basis.abs().max()/2)
                self.logger.log_image("val/gt_spatial_bases", [wandb.Image(grid, caption="Estimated Images")])

                grid = self.prepare_images(spatial_basis.abs(), gt_spatial_basis.abs().max()/2)
                self.logger.log_image("val/zero_filled", [wandb.Image(grid, caption="Zero filled images")])
                
                # [b, t, s, h, w]
                plot_sense = sense[0, 0, :, :, :].unsqueeze(1)

                grid = make_grid(plot_sense.abs())
                self.logger.log_image("val/sense_maps", [wandb.Image(grid, caption="sense")])

                grid = make_grid(plot_sense.angle())
                self.logger.log_image("train/sense_maps_angle", [wandb.Image(grid, caption="sense")])
                
        return loss
        
    def test_step(self, batch, batch_index): 
        undersampled, fully_sampled, sense = batch
        
        spatial_basis, _ = self.estimate_inital_bases(undersampled, sense, undersampled != 0)
        spatial_basis = view_as_real(spatial_basis)
        spatial_basis, mean, std = self.norm(spatial_basis)
        output = self.model(spatial_basis)
        denoised_spatial = spatial_basis + self.unnorm(output, mean, std)
        
        fully_sampled_image = (ifft_2d_img(fully_sampled)* sense.conj()).sum(2) / (sense.conj() * sense + 1e-6).sum(2)
        _, gt_spatial_basis = self.get_singular_vectors(fully_sampled_image)
        gt_spatial_basis = view_as_real(gt_spatial_basis.resolve_conj())

        ssim = metrics.calculate_ssim(denoised_spatial.abs(), gt_spatial_basis.abs(), self.device)
        ssim_loss = 1 - ssim
        loss = self.loss_fn(denoised_spatial, gt_spatial_basis)

        self.log('train/ssim_loss', ssim_loss, on_step=True, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        self.log('train/l1_loss', loss, on_step=True, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)

        loss = ssim_loss + loss
        return loss
        
    def forward(self, data):
        return self.model(data)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def rss(self, data):
        return ifft_2d_img(data).abs().pow(2).sum(2).sqrt()
    
    def prepare_images(self, imgs, max_val):
        imgs = imgs[0, :, :, :].unsqueeze(1)
        grid = make_grid(imgs, normalize=True, value_range=(0, max_val), nrow=3)
        return grid

    def estimate_inital_bases(self, reference_k, sense_maps, mask):
        masked_k = self.get_center_masked_k_space(reference_k) 
        masked_k = (ifft_2d_img(masked_k) * sense_maps.conj()).sum(2) / (sense_maps * sense_maps.conj() + 1e-6).sum(2)
        masked_k[torch.isnan(masked_k)] = 0
        temporal_basis, spatial_basis = self.get_singular_vectors(masked_k)
        cg_spatial = cg_data_consistency_R(iterations=4, lambda_reg=1).to(spatial_basis.device)

        spatial_basis = cg_spatial(reference_k, torch.zeros_like(spatial_basis, requires_grad=False, device=spatial_basis.device), sense_maps, temporal_basis, mask)
        return spatial_basis, temporal_basis

    def get_singular_vectors(self, data):
        b, t, h, w = data.shape

        temporal_basis, sv, spatial_basis = torch.linalg.svd(data.view(b, t, h*w), full_matrices=False, driver='gesvdj')
        components = 3 #(singular_values > singular_values[0]*self.singular_cuttoff).numel()
        temporal_basis = temporal_basis[:, :, :components]
        spatial_basis = spatial_basis[:, :components, :] * sv[:, :components].unsqueeze(-1)
        spatial_basis = spatial_basis.view(b, components, h, w)
        
        return temporal_basis, spatial_basis
    
    def get_center_masked_k_space(self, k_space):
        center_mask = torch.zeros_like(k_space, dtype=torch.bool)
        center_y, center_x = center_mask.shape[-2:]
        center_mask[:, :, :, center_y//2-8:center_y//2+8, center_x//2-8:center_x//2+8] = 1
        masked_k = k_space * center_mask
        return masked_k

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



###############################################################################
############# HELPER FUNCTIONS ################################################
###############################################################################

fft_2d_img = lambda x, axes=[-1, -2]: fftshift(fft2(ifftshift(x, dim=axes), dim=axes, norm='ortho'), dim=axes) 
ifft_2d_img = lambda x, axes=[-1, -2]: ifftshift(ifft2(fftshift(x, dim=axes), dim=axes, norm='ortho'), dim=axes) 

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

    


import numpy as np
import torch 

fft  = lambda x, ax : torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=ax), dim=ax, norm='ortho'), dim=ax) 
ifft = lambda X, ax : torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(X, dim=ax), dim=ax, norm='ortho'), dim=ax) 
fft_np  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
ifft_np = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

def espirit(X, k, r, t, c, device):
    """
    Derives the ESPIRiT operator.

    Arguments:
      X: Multi channel k-space data. Expected dimensions are (sx, sy, nc), where (sx, sy) are volumetric 
         dimensions and (nc) is the channel dimension.
      k: Parameter that determines the k-space kernel size. If X has dimensions (1, 256, 256, 8), then the kernel 
         will have dimensions (1, k, k, 8)
      r: Parameter that determines the calibration region size. If X has dimensions (1, 256, 256, 8), then the 
         calibration region will have dimensions (1, r, r, 8)
      t: Parameter that determines the rank of the auto-calibration matrix (A). Singular values below t times the
         largest singular value are set to zero.
      c: Crop threshold that determines eigenvalues "=1".
    Returns:
      maps: This is the ESPIRiT operator. It will have dimensions (sx, sy, sz, nc, nc) with (sx, sy, sz, :, idx)
            being the idx'th set of ESPIRiT maps.
    """

    sx = X.shape[1]
    sy = X.shape[2]
    nc = X.shape[3]

    sxt = (sx//2-r//2, sx//2+r//2) if (sx > 1) else (0, 1)
    syt = (sy//2-r//2, sy//2+r//2) if (sy > 1) else (0, 1)

    # Extract calibration region.    
    C = X[:, sxt[0]:sxt[1], syt[0]:syt[1], :]

    A = torch.nn.functional.unfold(C.permute(0, 3, 1, 2), kernel_size=k)
    A = A.permute(0, 2, 1)

    # Take the Singular Value Decomposition.
    _, S, V = torch.linalg.svd(A, full_matrices=False)


    V = V.conj().transpose(-1, -2)

    # Select kernels.
    n = torch.max(torch.sum((S >= t * S[:, [0]]), dim=1))
    
    del S, C, A

    V = V[:, :, 0:n]

    # Reshape into k-space kernel, flips it and takes the conjugate
        
    kernels = torch.reshape(V, (V.shape[0], nc, k , k, n.item()))
    kernels = kernels.permute(0, 2, 3, 1, 4)
    pad_x = sx - k
    pad_y = sy -k
    kernels = torch.nn.functional.pad(kernels, (0, 0, 0, 0, pad_y//2, pad_y//2, pad_x//2, pad_x//2, 0, 0))

    # Take the iucfft
    axes = (1, 2)
    kernels.data = torch.flip(kernels.conj(), axes)
    kerimgs = fft(kernels, axes) * torch.sqrt(torch.tensor(sx * sy))/torch.sqrt(torch.tensor(k**2))
    del kernels, V

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    driver = None
    if device == 'cuda': 
        driver = 'gesvda'
    try:
        u, _, _ = torch.linalg.svd(kerimgs, full_matrices=False, driver=driver)
    except torch.cuda.OutOfMemoryError:
        u1, _, _ = torch.linalg.svd(kerimgs[:, :kerimgs.shape[1]//2, :, :, :], full_matrices=False, driver=driver)
        u2, _, _ = torch.linalg.svd(kerimgs[:, kerimgs.shape[1]//2:, :, :, :], full_matrices=False, driver=driver)
        u = torch.cat((u1, u2), dim=1)

    maps = u[..., 0]

    return maps

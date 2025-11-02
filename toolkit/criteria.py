import torch

def psnr(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2, dim=(1,2,3)) + eps
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse))

def ssim_simple(pred, target, C1=0.01**2, C2=0.03**2):
    mu_x = torch.mean(pred, dim=(2,3), keepdim=True)
    mu_y = torch.mean(target, dim=(2,3), keepdim=True)
    sigma_x = torch.var(pred, dim=(2,3), unbiased=False, keepdim=True)
    sigma_y = torch.var(target, dim=(2,3), unbiased=False, keepdim=True)
    sigma_xy = torch.mean((pred - mu_x) * (target - mu_y), dim=(2,3), keepdim=True)
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return (num / (den + 1e-8)).squeeze().mean(dim=1)
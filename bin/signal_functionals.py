import torch
import torch.nn.functional as F
from scipy.signal import correlate as scipy_correlate

def lt_transform(signal, eps=1e-16):
    return 1 - (signal / (torch.min(signal, dim=-1, keepdim=True)[0] + eps))

def MMIS(signal, eps=1e-16):
    signal = signal / (torch.max(signal, dim=-1, keepdim=True)[0] + eps)
    signal = lt_transform(signal)
    signal = signal / (1 + signal)
    return signal

def max_transform(signal, eps=1e-16):
    return signal / (torch.max(signal, dim=-1, keepdim=True)[0] + eps)

def minmax_transform(signal, eps=1e-16):
    mn = torch.min(signal, dim=-1, keepdim=True)[0]
    mx = torch.max(signal, dim=-1, keepdim=True)[0]
    return (signal - mn) / (mx - mn)

def triangular_kernel(lags, l, device):
    if l == 0:
        kernel = torch.zeros_like(lags).to(device=device)
        kernel[lags == 0] = 1
    else:
        kernel = torch.maximum(1 - torch.abs(lags) / l, torch.zeros_like(lags)).to(device=device)
    return kernel

def rwp(calc, obs, weights=None, transform=None, device='cpu'):
    # Making sure that data arrays align
    if not isinstance(calc, torch.Tensor):
        calc = torch.tensor(calc, dtype=torch.float64)
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float64)
    assert calc.shape == obs.shape

    # Push to device
    calc = calc.to(device=device)
    obs = obs.to(device=device)
    
    # Transform
    if transform is not None:
        calc = transform(calc)
        obs = transform(obs)
    
    # Making sure that weights are aligned
    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float64).to(device=device)
        else:
            weights.to(device=device)
        assert weights.shape == calc.shape
    else:
        weights = torch.ones_like(calc).to(device=device)

    # Return Rwp
    return torch.sqrt(torch.sum(weights * (obs - calc)**2, dim=-1) / torch.sum(weights * obs**2, dim=-1))

def s12(data1, data2, l=0, transform=lt_transform, device='cpu'):
    # Making sure that data arrays align
    if not isinstance(data1, torch.Tensor):
        data1 = torch.tensor(data1, dtype=torch.float64)
    if not isinstance(data2, torch.Tensor):
        data2 = torch.tensor(data2, dtype=torch.float64)
    # Push to device
    data1 = data1.to(device=device)
    data2 = data2.to(device=device)

    if len(data1.shape) == 1:
            data1.unsqueeze_(0)
    if len(data2.shape) == 1:
            data2.unsqueeze_(0)

    assert data1.shape == data2.shape

    # Apply transform
    if transform is not None:
        data1 = transform(data1)
        data2 = transform(data2)

    # Correlations
    lags = torch.arange(-data1.shape[-1] + 1, data1.shape[-1]).to(device=device)

    cross_corr = F.conv1d(data1.unsqueeze(1), data2.unsqueeze(1), padding=data1.shape[-1]-1).mean(dim=1, keepdim=False)
    auto_corr1 = F.conv1d(data1.unsqueeze(1), data1.unsqueeze(1), padding=data1.shape[-1]-1).mean(dim=1, keepdim=False)
    auto_corr2 = F.conv1d(data2.unsqueeze(1), data2.unsqueeze(1), padding=data2.shape[-1]-1).mean(dim=1, keepdim=False)

    # Kernel
    weight_kernel = triangular_kernel(lags, l, device=device)

    # Correlation coefs
    c12 = torch.sum(weight_kernel * cross_corr, dim=-1)
    c11 = torch.sum(weight_kernel * auto_corr1, dim=-1)
    c22 = torch.sum(weight_kernel * auto_corr2, dim=-1)

    # S12
    s12 = c12 / torch.sqrt(c11 * c22)

    return s12

import os

import torch
import torch.nn.functional as F


def get_device():
    """NCM_DEVICE if set, else cuda, then Apple Silicon mps, then cpu."""
    dev = os.environ.get("NCM_DEVICE")
    if dev:
        return dev
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def float64_device(device):
    """mps has no float64; float64 work for an mps tensor runs on cpu instead."""
    return "cpu" if torch.device(device).type == "mps" else device


_POOL_CACHE = {}


def area_interpolate(x, size):
    """F.interpolate(x, size=(size,), mode="area") for (B, C, L) input.

    mps only implements adaptive average pooling when L is divisible by size, so there it
    is computed as a matmul with the same bins adaptive_avg_pool1d uses:
    [floor(i*L/size), ceil((i+1)*L/size)).
    """
    L = x.shape[-1]
    if x.device.type != "mps" or L % size == 0:
        return F.interpolate(x, size=(size,), mode="area")
    key = (L, size, x.dtype, x.device)
    W = _POOL_CACHE.get(key)
    if W is None:
        W = torch.zeros(L, size, dtype=torch.float32)
        for i in range(size):
            s, e = (i * L) // size, -((-(i + 1) * L) // size)
            W[s:e, i] = 1.0 / (e - s)
        W = _POOL_CACHE[key] = W.to(device=x.device, dtype=x.dtype)
    return x @ W

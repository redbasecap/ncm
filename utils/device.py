import os

import torch
import torch.nn.functional as F


_MPS_OK = None


def mps_usable():
    """mps can report available yet fail to allocate (e.g. some virtualized Macs); try a real op."""
    global _MPS_OK
    if _MPS_OK is None:
        _MPS_OK = False
        if torch.backends.mps.is_available():
            try:
                x = torch.randn(256, 256, device="mps")
                _MPS_OK = bool(torch.isfinite((x @ x).sum()).item())
            except RuntimeError as e:
                print(f"[device] mps unusable ({str(e).splitlines()[0][:120]}); using cpu", flush=True)
    return _MPS_OK


def get_device():
    """NCM_DEVICE if set, else cuda, then Apple Silicon mps (if it really works), then cpu."""
    dev = os.environ.get("NCM_DEVICE")
    if dev:
        return dev
    if torch.cuda.is_available():
        return "cuda"
    if mps_usable():
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


def _empty_cache():
    if mps_usable():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def retry_on_oom(fn, batch_size):
    """fn(batch_size), halving batch_size on out-of-memory so small-memory Macs work with defaults."""
    while True:
        try:
            return fn(batch_size)
        except (RuntimeError, torch.OutOfMemoryError) as e:
            if "out of memory" not in str(e).lower() or batch_size <= 1:
                raise
            _empty_cache()
            batch_size //= 2
            print(f"[oom] retrying with batch size {batch_size}", flush=True)


class _ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        # .contiguous() is a no-op for size-1 dims with odd strides, so force a fresh standard layout.
        return torch.empty(g.shape, dtype=g.dtype, device=g.device).copy_(g)


def contiguous_grad(x):
    """Identity whose backward makes the incoming gradient contiguous.

    The mps convolution backward aborts ("Weights tensor and ndArray input channel mismatch") when the
    gradient reaching a Conv1d with output length 1 is non-contiguous, which happens when the conv
    output is transposed afterwards. Only applied on mps.
    """
    return _ContiguousGrad.apply(x) if x.device.type == "mps" and x.requires_grad else x

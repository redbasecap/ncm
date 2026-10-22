"""Apple Silicon (mps) checks: run with `python -m pytest tests/` or `python tests/test_mps.py`."""
import glob
import os
import sys

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.device import area_interpolate, float64_device, get_device, mps_usable, retry_on_oom

MPS = mps_usable()


def _level_pairs():
    pairs = set()
    for f in glob.glob(os.path.join(ROOT, "config", "eval", "*.yaml")):
        vq = OmegaConf.load(f).tokenizer.vqvae
        lv = list(vq.vector_quantizer_config.levels.manual_levels)
        for i, l in enumerate(lv):
            pairs.add((lv[-1], l))
            if i + 1 < len(lv):
                pairs.add((lv[-1], lv[i + 1]))
    return sorted(pairs)


@pytest.mark.parametrize("L,size", _level_pairs())
def test_area_interpolate_matches_reference_cpu(L, size):
    x = torch.randn(3, 16, L)
    torch.testing.assert_close(area_interpolate(x, size), F.interpolate(x, size=(size,), mode="area"))


@pytest.mark.skipif(not MPS, reason="needs a working Apple Silicon mps device")
@pytest.mark.parametrize("L,size", _level_pairs())
def test_area_interpolate_mps_matches_cpu(L, size):
    x = torch.randn(3, 16, L)
    ref = F.interpolate(x, size=(size,), mode="area")
    torch.testing.assert_close(area_interpolate(x.to("mps"), size).cpu(), ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not MPS, reason="needs a working Apple Silicon mps device")
def test_linear_upsample_mps_matches_cpu():
    x = torch.randn(3, 16, 13)
    ref = F.interpolate(x, size=(89,), mode="linear")
    torch.testing.assert_close(F.interpolate(x.to("mps"), size=(89,), mode="linear").cpu(), ref, rtol=1e-5, atol=1e-5)


def test_device_selection(monkeypatch):
    monkeypatch.setenv("NCM_DEVICE", "cpu")
    assert get_device() == "cpu"
    monkeypatch.delenv("NCM_DEVICE")
    assert get_device() in ("cuda", "mps", "cpu")
    assert float64_device("mps") == "cpu" and float64_device("cpu") == "cpu"


def test_retry_on_oom_halves_batch_size():
    seen = []

    def fn(bs):
        seen.append(bs)
        if bs > 4:
            raise RuntimeError("MPS backend out of memory (simulated)")
        return bs

    assert retry_on_oom(fn, 32) == 4 and seen == [32, 16, 8, 4]
    with pytest.raises(RuntimeError, match="unrelated"):
        retry_on_oom(lambda bs: (_ for _ in ()).throw(RuntimeError("unrelated")), 8)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

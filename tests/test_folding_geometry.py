"""Geometry checks for the protein folding prototype (folding/)."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from folding.geometry import ca_internal, dihedral, lddt_ca, rebuild_ca, rmsd, tm_score_kabsch

SAMPLE = os.path.join(ROOT, "data", "cath", "sample.jsonl")


def _resolved_ca(n=20, min_len=40):
    out = []
    for line in open(SAMPLE):
        CA = np.array(json.loads(line)["coords"]["CA"], dtype=np.float64)
        if len(CA) >= min_len and not np.isnan(CA).any() and (np.linalg.norm(np.diff(CA, axis=0), axis=-1) < 4.2).all():
            out.append(CA)
        if len(out) == n:
            break
    return out


def test_dihedral_sign_convention():
    # Right-handed: looking down b->c, d rotated +90 deg clockwise from a gives +90.
    a, b, c, d = np.array([1.0, 0, 0]), np.zeros(3), np.array([0, 0, 1.0]), np.array([0, 1.0, 1.0])
    assert np.isclose(np.rad2deg(dihedral(a, b, c, d)), 90.0)


def test_rebuild_inverts_internal_coordinates():
    rng = np.random.default_rng(0)
    L = 50
    theta, tau = rng.uniform(1.4, 2.4, L), rng.uniform(-np.pi, np.pi, L)
    X = rebuild_ca(theta, tau)
    th2, ta2 = ca_internal(X)
    assert np.allclose(th2[1:-1], theta[1:-1]) and np.allclose(ta2[1:-2], tau[1:-2])
    assert np.allclose(np.linalg.norm(np.diff(X, axis=0), axis=-1), 3.806)


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="needs data/cath/sample.jsonl")
def test_true_angles_rebuild_true_structure():
    chains = _resolved_ca()
    assert chains, "no fully resolved chains in the sample"
    for CA in chains:
        X = rebuild_ca(*ca_internal(CA))
        assert rmsd(X, CA) < 1.0, len(CA)  # only CA-CA distance variation is lost
        assert tm_score_kabsch(X, CA) > 0.95 and lddt_ca(X, CA) > 0.95


def test_metrics_identity_and_noise():
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.normal(size=(60, 3)), axis=0) * 2.0
    assert rmsd(X, X) < 1e-8 and np.isclose(tm_score_kabsch(X, X), 1.0) and np.isclose(lddt_ca(X, X), 1.0)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert rmsd(X @ R.T + 5.0, X) < 1e-8  # rigid-motion invariant
    assert tm_score_kabsch(X + rng.normal(scale=3.0, size=X.shape), X) < 0.8

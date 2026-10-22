"""CA-trace geometry in numpy (float64, cpu): internal coordinates, NeRF rebuild, structure metrics.

A chain is represented by one (theta, tau) pair per residue: the virtual CA bond angle and the
virtual CA dihedral. With a fixed 3.806 A CA-CA distance this rebuilds real chains to ~0.1-0.4 A
RMSD, whereas phi/psi with ideal bond geometry drifts by several A over 100+ residues.
"""
import numpy as np


def dihedral(a, b, c, d):
    """Signed dihedral angle (radians) for arrays of points with shape (..., 3)."""
    b0, b1, b2 = a - b, c - b, d - c
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, -1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, -1, keepdims=True) * b1n
    x = np.sum(v * w, -1)
    y = np.sum(np.cross(b1n, v) * w, -1)
    return np.arctan2(y, x)


def angle(a, b, c):
    """Bond angle a-b-c (radians)."""
    u, v = a - b, c - b
    cos = np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.arccos(np.clip(cos, -1.0, 1.0))


CA_CA = 3.806  # median consecutive CA-CA distance (trans peptides) in CATH 4.2


def ca_internal(CA):
    """CA-trace internal coordinates: theta (L,) virtual bond angle at i, tau (L,) virtual dihedral
    of (i-1, i, i+1, i+2). theta is defined for 1..L-2 and tau for 1..L-3; elsewhere nan."""
    L = len(CA)
    theta, tau = np.full(L, np.nan), np.full(L, np.nan)
    if L >= 3:
        theta[1:-1] = angle(CA[:-2], CA[1:-1], CA[2:])
    if L >= 4:
        tau[1:-2] = dihedral(CA[:-3], CA[1:-2], CA[2:-1], CA[3:])
    return theta, tau


def _place(a, b, c, bond, ang, tor):
    """NeRF: position of d with |cd| = bond, angle(b, c, d) = ang, dihedral(a, b, c, d) = tor."""
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.stack([bc, np.cross(n, bc), n], axis=-1)
    d2 = np.array([-bond * np.cos(ang), bond * np.sin(ang) * np.cos(tor), bond * np.sin(ang) * np.sin(tor)])
    return c + m @ d2


def rebuild_ca(theta, tau, bond=CA_CA):
    """CA coordinates (L, 3) from ca_internal() angles, in a canonical frame.

    Uses theta[1..L-2] and tau[1..L-3]; nan entries there fall back to helix-like defaults.
    """
    L = len(theta)
    theta = np.where(np.isnan(theta), np.deg2rad(91.0), theta)
    tau = np.where(np.isnan(tau), np.deg2rad(50.0), tau)
    X = np.zeros((max(L, 1), 3))
    if L > 1:
        X[1] = [bond, 0.0, 0.0]
    if L > 2:
        X[2] = X[1] + bond * np.array([-np.cos(theta[1]), np.sin(theta[1]), 0.0])
    for i in range(3, L):
        X[i] = _place(X[i - 3], X[i - 2], X[i - 1], bond, theta[i - 1], tau[i - 2])
    return X[:L]


def kabsch(P, Q):
    """P rotated and translated onto Q (both (L, 3)), minimizing RMSD."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    s = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, s])
    return Pc @ U @ D @ Vt + Q.mean(0)


def rmsd(P, Q):
    """CA RMSD after optimal superposition."""
    return float(np.sqrt(np.mean(np.sum((kabsch(P, Q) - Q) ** 2, -1))))


def tm_score_kabsch(P, Q):
    """TM-score of P against reference Q under the RMSD-optimal superposition.

    This is a lower bound on the TM-align TM-score, which searches for the superposition
    that maximizes TM-score itself.
    """
    L = len(Q)
    d0 = max(1.24 * np.cbrt(max(L, 19) - 15) - 1.8, 0.5)
    d = np.linalg.norm(kabsch(P, Q) - Q, axis=-1)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


def lddt_ca(P, Q, cutoff=15.0):
    """Superposition-free CA lDDT (thresholds 0.5/1/2/4 A) of P against reference Q."""
    dq = np.linalg.norm(Q[:, None] - Q[None], axis=-1)
    dp = np.linalg.norm(P[:, None] - P[None], axis=-1)
    mask = (dq < cutoff) & ~np.eye(len(Q), dtype=bool)
    diff = np.abs(dp - dq)[mask]
    if diff.size == 0:
        return float("nan")
    return float(np.mean([(diff < t).mean() for t in (0.5, 1.0, 2.0, 4.0)]))

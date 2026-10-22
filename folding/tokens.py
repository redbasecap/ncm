"""Structure tokens: k-means clusters of per-residue CA (theta, tau) angles."""
import numpy as np

from folding.geometry import ca_internal, rebuild_ca


def features(theta, tau):
    return np.stack([np.cos(theta), np.sin(theta), np.cos(tau), np.sin(tau)], -1)


class AngleTokenizer:
    """K angle clusters plus TERM (residues whose theta/tau is undefined: chain ends) and PAD."""

    def __init__(self, centroids):
        self.centroids = np.asarray(centroids, dtype=np.float64)  # (K, 4) feature space
        self.K = len(self.centroids)
        self.TERM, self.PAD = self.K, self.K + 1
        self.vocab_size = self.K + 2
        c = self.centroids
        self.theta = np.arctan2(c[:, 1], c[:, 0])
        self.tau = np.arctan2(c[:, 3], c[:, 2])

    @classmethod
    def fit(cls, ca_list, K, iters=40, max_points=400_000, seed=0):
        rng = np.random.default_rng(seed)
        X = np.concatenate([features(*ca_internal(ca)) for ca in ca_list])
        X = X[~np.isnan(X).any(-1)]
        if len(X) > max_points:
            X = X[rng.choice(len(X), max_points, replace=False)]
        C = X[rng.choice(len(X), K, replace=False)]
        for _ in range(iters):
            a = cls._assign(X, C)
            for k in range(K):
                m = a == k
                C[k] = X[m].mean(0) if m.any() else X[rng.integers(len(X))]
        return cls(C)

    @staticmethod
    def _assign(X, C):
        return ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(-1)

    def encode(self, ca):
        f = features(*ca_internal(ca))
        ok = ~np.isnan(f).any(-1)
        out = np.full(len(ca), self.TERM, dtype=np.int64)
        if ok.any():
            out[ok] = self._assign(f[ok], self.centroids)
        return out

    def decode(self, tokens):
        """CA coordinates from tokens; TERM/PAD fall back to rebuild_ca defaults."""
        t = np.asarray(tokens)
        ang = t < self.K
        theta, tau = np.full(len(t), np.nan), np.full(len(t), np.nan)
        theta[ang], tau[ang] = self.theta[t[ang]], self.tau[t[ang]]
        return rebuild_ca(theta, tau)

    def state_dict(self):
        return {"centroids": self.centroids}

"""Symmetric-positive-definite (SPD) manifold primitives — pure numpy/scipy.

The flagship "Geometry of Calm" pipeline represents each physiological window by an
SPD covariance matrix and works with the *geometry* of these matrices rather than
their raw entries. This module provides the small set of Riemannian operations that
need (all on the affine-invariant / log-Euclidean metric):

  * matrix functions on SPD inputs (sqrt, inverse-sqrt, log, exp) via eigh,
  * the affine-invariant geodesic distance,
  * the Frechet (geometric) mean of a set of SPD matrices,
  * tangent-space projection at a reference (the feature map fed to a classifier),
  * Riemannian re-centering / whitening (the domain-adaptation step), and
  * the Bures-Wasserstein 2-Wasserstein distance between Gaussians (the optimal-
    transport effect size).

Everything is deterministic and dependency-light so it slots into the project's
leak-free evaluation without adding heavy libraries.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------
def shrink_cov(x: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Ledoit-Wolf-style shrinkage covariance of an (n_channels, n_samples) array.

    Shrinking toward a scaled identity guarantees a well-conditioned SPD matrix even
    for short windows or few channels (essential for the 2-channel custom data).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    c = (x @ x.T) / max(1, n - 1)
    d = c.shape[0]
    mu = np.trace(c) / d
    return (1.0 - shrink) * c + shrink * mu * np.eye(d)


# ---------------------------------------------------------------------------
# Matrix functions on SPD inputs
# ---------------------------------------------------------------------------
def _eig_apply(c: np.ndarray, fn) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    w, v = np.linalg.eigh((c + c.T) / 2.0)
    w = np.clip(w, 1e-12, None)
    return (v * fn(w)) @ v.T


def sqrtm_spd(c):       return _eig_apply(c, np.sqrt)
def invsqrtm_spd(c):    return _eig_apply(c, lambda w: 1.0 / np.sqrt(w))
def logm_spd(c):        return _eig_apply(c, np.log)
def expm_spd(c):        return _eig_apply(c, np.exp)
def powm_spd(c, p):     return _eig_apply(c, lambda w: np.power(w, p))


# ---------------------------------------------------------------------------
# Distances and means
# ---------------------------------------------------------------------------
def airm_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Affine-invariant Riemannian distance between two SPD matrices."""
    a_is = invsqrtm_spd(a)
    m = a_is @ b @ a_is
    w = np.clip(np.linalg.eigvalsh((m + m.T) / 2.0), 1e-12, None)
    return float(np.sqrt(np.sum(np.log(w) ** 2)))


def frechet_mean(covs: np.ndarray, max_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Geometric (Frechet) mean of SPD matrices on the affine-invariant manifold.

    Iterative gradient descent in the tangent space (Karcher flow), initialised at the
    arithmetic mean. Converges in a handful of steps for well-conditioned inputs.
    """
    covs = np.asarray(covs, dtype=float)
    mean = covs.mean(axis=0)
    for _ in range(max_iter):
        m_is = invsqrtm_spd(mean)
        m_sq = sqrtm_spd(mean)
        # mean of logs in the whitened tangent space
        tang = np.mean([logm_spd(m_is @ c @ m_is) for c in covs], axis=0)
        mean = m_sq @ expm_spd(tang) @ m_sq
        mean = (mean + mean.T) / 2.0
        if np.linalg.norm(tang) < tol:
            break
    return mean


# ---------------------------------------------------------------------------
# Tangent space (the feature map)
# ---------------------------------------------------------------------------
def _upper_vec(s: np.ndarray) -> np.ndarray:
    """Vectorise a symmetric matrix, scaling off-diagonals by sqrt(2) (isometry)."""
    d = s.shape[0]
    iu = np.triu_indices(d, 1)
    out = np.empty(d + iu[0].size)
    out[:d] = np.diag(s)
    out[d:] = np.sqrt(2.0) * s[iu]
    return out


def tangent_vectors(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Project SPD matrices to the tangent space at ``ref`` and vectorise.

    T_i = upper_vec( logm( ref^{-1/2} C_i ref^{-1/2} ) ). The resulting Euclidean
    vectors can be handed to any standard classifier; distances in this space locally
    approximate the manifold geodesic distance.
    """
    r_is = invsqrtm_spd(ref)
    return np.stack([_upper_vec(logm_spd(r_is @ c @ r_is)) for c in covs])


def recenter(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Riemannian re-centering: whiten a domain so its geometric mean maps to identity.

    C_i -> ref^{-1/2} C_i ref^{-1/2}. Applying this to each domain with that domain's
    own Frechet mean removes a multiplicative (device/gain/montage) shift — the core
    of the device-invariant transfer step.
    """
    r_is = invsqrtm_spd(ref)
    return np.stack([r_is @ c @ r_is for c in covs])


# ---------------------------------------------------------------------------
# Optimal transport: Bures-Wasserstein distance between Gaussians
# ---------------------------------------------------------------------------
def bures_wasserstein(mu1, cov1, mu2, cov2) -> dict:
    """Squared 2-Wasserstein distance between two Gaussians, split into its parts.

        W2^2 = ||mu1 - mu2||^2 + Tr( cov1 + cov2 - 2 (cov1^{1/2} cov2 cov1^{1/2})^{1/2} )

    The mean term measures the shift of the central physiological state; the covariance
    (Bures) term measures the change in its dispersion/shape. Returns both so the effect
    size is interpretable, not just a single opaque number.
    """
    mu1, mu2 = np.asarray(mu1, float), np.asarray(mu2, float)
    cov1, cov2 = np.asarray(cov1, float), np.asarray(cov2, float)
    mean_term = float(np.sum((mu1 - mu2) ** 2))
    s1 = sqrtm_spd(cov1)
    inner = sqrtm_spd(s1 @ cov2 @ s1)
    cov_term = float(np.trace(cov1 + cov2 - 2.0 * inner))
    cov_term = max(cov_term, 0.0)  # guard tiny negative from numerical error
    w2sq = mean_term + cov_term
    return dict(w2=float(np.sqrt(max(w2sq, 0.0))), w2_sq=float(w2sq),
                mean_term=mean_term, cov_term=cov_term)


def gaussian_fit(x: np.ndarray, shrink: float = 0.05):
    """Mean and shrinkage covariance of a set of feature vectors (rows = samples)."""
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=0)
    xc = x - mu
    c = (xc.T @ xc) / max(1, x.shape[0] - 1)
    d = c.shape[0]
    mtr = np.trace(c) / d
    c = (1.0 - shrink) * c + shrink * mtr * np.eye(d)
    return mu, c

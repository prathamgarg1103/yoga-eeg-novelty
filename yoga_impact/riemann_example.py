"""Worked example — trace ONE real window through the Geometry-of-Calm math.

Run:  .venv/Scripts/python.exe -m yoga_impact.riemann_example

Every intermediate is printed so a reader can verify Sections 3-5 of
docs/FLAGSHIP_Geometry_of_Calm.md on real numbers: virtual channels -> window
covariance C -> Frechet reference G -> tangent vector T -> Bures-Wasserstein W2.
Nothing here is used by the pipeline; it exists purely to make the math inspectable.
"""
from __future__ import annotations

import numpy as np

from yoga_impact import config
from yoga_impact import riemann_spd as R
from yoga_impact.clean_custom import load_clean_recordings
from yoga_impact.riemann import _virtual_channels, build_eeg_covs


def main() -> None:
    np.set_printoptions(precision=3, suppress=True)
    recs = load_clean_recordings()
    rec = next(r for r in recs if r.state == config.POSITIVE_CLASS)   # first Relaxed
    print(f"recording: subject={rec.subject}  state={rec.state}  device={rec.device}  "
          f"fs={rec.fs:.0f} Hz  samples={rec.data.shape[1]}")

    bands = config.RIEMANN_BANDS
    names = [f"{ch}-{b}" for ch in rec.channels for b in bands]
    V = _virtual_channels(rec, bands)                                 # (6, N)
    print(f"virtual channels ({len(names)}): {names}")

    win = int(config.WIN_SEC * rec.fs)
    C = R.shrink_cov(V[:, :win], config.RIEMANN_SHRINKAGE)           # first 8 s window
    print(f"\n[1] window covariance C  ({C.shape[0]}x{C.shape[1]}, shrinkage="
          f"{config.RIEMANN_SHRINKAGE}):")
    print(C)
    eig = np.linalg.eigvalsh(C)
    print("    eigenvalues(C):", eig, " -> all > 0 (SPD):", bool(np.all(eig > 0)))

    covs, _band, meta = build_eeg_covs(
        recs, label_fn=lambda r: r.state == config.POSITIVE_CLASS)
    G = R.frechet_mean(covs)
    print(f"\n[2] Frechet (geometric) mean G over all {len(covs)} custom windows:")
    print("    eigenvalues(G):", np.linalg.eigvalsh(G))
    d0 = R.airm_distance(C, G)
    print(f"    geodesic distance d(C, G) = {d0:.4f}")

    T = R.tangent_vectors(np.stack([C]), G)[0]
    print(f"\n[3] tangent vector T = upper_vec( log( G^-1/2 C G^-1/2 ) ), dim={T.size}:")
    print("   ", T)
    print(f"    ||T|| = {np.linalg.norm(T):.4f}  (equals d(C,G) = {d0:.4f}: the tangent map "
          f"is an isometry)")

    Tan = R.tangent_vectors(covs, G)
    lab = meta["state"].to_numpy() == config.POSITIVE_CLASS
    mu1, S1 = R.gaussian_fit(Tan[lab], config.OT_GAUSS_SHRINK)
    mu2, S2 = R.gaussian_fit(Tan[~lab], config.OT_GAUSS_SHRINK)
    bw = R.bures_wasserstein(mu1, S1, mu2, S2)
    print(f"\n[4] Bures-Wasserstein W2 between Relaxed (n={int(lab.sum())}) and "
          f"Concentrated (n={int((~lab).sum())}) in tangent space:")
    print(f"    mean term ||mu1-mu2||^2  = {bw['mean_term']:.3f}")
    print(f"    shape term Tr(S1+S2-2..) = {bw['cov_term']:.3f}")
    print(f"    W2 = sqrt(mean + shape)  = {bw['w2']:.3f}")


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    main()

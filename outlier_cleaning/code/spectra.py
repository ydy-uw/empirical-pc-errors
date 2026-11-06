"""Centred dual spectra theta_1..theta_62 of every treated panel (aggregates only).

    python outlier_cleaning/code/spectra.py <method> [<method> ...]

Writes experiments/<method>/results/spectra_<setting>.npz with arrays formation_date,
n_assets and theta (956 x 62).  The bias of Proposition B needs A_1..A_k for all k
directions, which the per-panel CSV files do not keep.
"""
from __future__ import annotations
import sys
import numpy as np
from oc_common import EXP, P_GRID
import panels, treatments


def main(methods):
    for m in methods:
        for s in treatments.METHODS[m][1]:
            fd_, P_, th_ = [], [], []
            for P in P_GRID:
                d = panels.load(P)
                for fd in sorted(d):
                    Xt, _, _ = treatments.apply(m, s, d[fd]["X"], d[fd])
                    A = Xt - Xt.mean(axis=0)
                    sv = np.linalg.svd(A, compute_uv=False)
                    N, p = A.shape
                    fd_.append(np.datetime64(fd, "D")); P_.append(P)
                    th_.append((sv ** 2 / (N * p))[:N - 1])
            np.savez(EXP / m / "results" / f"spectra_{s}.npz", formation_date=np.array(fd_),
                     n_assets=np.array(P_), theta=np.array(th_))
            print(m, s, "spectra written", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

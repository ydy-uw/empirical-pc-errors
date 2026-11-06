"""Co-movement condition of the day-variance proposition, computed with PCA leverages and the
leverage-corrected day variances (Section 6.2).  Same definition as heavy.fa's comove_ratio:
ratio = (tr(P Dt P)/k) / (tr(Dt)/(N-1)), Dt = M diag(d) M, M = I - 11'/N, P = U_k U_k' from the
clipped, demeaned 63-day panel, d_t the corrected day variance; condition holds if ratio >= 1.
Computed at k = 3 and at the month's skewness count, all 239 months, P = 3,000.
Output: rerun_20260928/work/figure_code/out/checks_20261011/comove_pca.csv
"""
import os, sys, pickle
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "rerun_20260928" / "work"
panels = pickle.load(open(WORK / "outlier_cleaning" / "_cache" / "panels_P3000.pkl", "rb"))
c = pd.read_csv(WORK / "outlier_cleaning" / "experiments" / "01_fixed_clip" / "results" / "panels_c050.csv", parse_dates=["formation_date"])
ks = c[c.n_assets == 3000].set_index("formation_date").k_skew
rows = []
for fd in sorted(ks.index):
    Y = np.clip(panels[pd.Timestamp(fd)]["X"], -0.5, 0.5); Y = Y - Y.mean(axis=0, keepdims=True)
    N = Y.shape[0]; U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    M = np.eye(N) - 1.0 / N
    row = dict(formation_date=fd, k_skew=int(ks.loc[fd]))
    for tag, k in (("k3", 3), ("kown", int(ks.loc[fd]))):
        E = Y - (U[:, :k] * s[:k]) @ Vt[:k]
        h = 1.0 / N + (U[:, :k] ** 2).sum(axis=1)
        d = (E ** 2).mean(axis=1) / (1 - h)
        Dt = M @ (d[:, None] * M)
        P = U[:, :k] @ U[:, :k].T
        row[f"ratio_{tag}"] = float((np.trace(P @ Dt @ P) / k) / (np.trace(Dt) / (N - 1)))
    rows.append(row)
d = pd.DataFrame(rows); out = WORK / "figure_code" / "out" / "checks_20261011"; d.to_csv(out / "comove_pca.csv", index=False)
for tag in ("k3", "kown"):
    print(tag, "share ratio>=1: %.3f (%d of %d); median ratio %.2f" % ((d[f"ratio_{tag}"] >= 1).mean(), int((d[f"ratio_{tag}"] >= 1).sum()), len(d), d[f"ratio_{tag}"].median()))

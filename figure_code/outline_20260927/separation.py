"""Assumptions 5-6 (distinct factor eigenvalues) on the clipped P = 3,000 panels: are the sample
directions stable from one month to the next, and does that depend on the eigenvalue gaps?

For each pair of consecutive formation months the two 63-day windows share about 42 days.  On the
stocks common to both universes, each month's clipped, demeaned panel gives sample directions
h_1, h_2, h_3 (unit P-vectors) and dual eigenvalues theta_1 >= theta_2 >= ...  Recorded per pair:
  ang_j        angle between h_j of the earlier and of the later month, j = 1, 2, 3 (degrees);
  span23       largest principal angle between span(h_2, h_3) of the two months;
  span123      largest principal angle between span(h_1, h_2, h_3) of the two months;
  ang2_in23    angle between the later h_2 and span(h_2, h_3) of the earlier month (ang3_in23 for h_3);
  ang2_in1234  angle between the later h_2 and span(h_1, ..., h_4) of the earlier month;
  gap12, gap23 relative gaps (theta_j - theta_{j+1}) / theta_j of the later month, and the
               smaller of the two months' gap23 (mingap23).
With well-separated eigenvalues and heavily overlapping windows each h_j should move little; when
theta_2 and theta_3 nearly coincide, h_2 and h_3 can rotate into each other while their span stays.

Inputs: the licensed CRSP extract through companion/code/rawpanel.py (read-only).
Output (aggregates only): figure_code/out/outline_20260927/separation.csv
Run from the project root:  OPENBLAS_NUM_THREADS=1 python figure_code/outline_20260927/separation.py
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "companion" / "code"))
import rawpanel  # noqa: E402

OUT = ROOT / "figure_code" / "out" / "outline_20260927"
P, N_OBS, CLIP = 3000, 63, 0.5


def directions(W, fd, cols):
    X = W.loc[pd.DatetimeIndex(rawpanel.window_dates(fd, N_OBS)), cols].to_numpy(float)
    Y = np.clip(X, -CLIP, CLIP); Y = Y - Y.mean(axis=0, keepdims=True)
    _, s, Vt = np.linalg.svd(Y, full_matrices=False)
    th = s ** 2 / ((N_OBS - 1) * Y.shape[1])
    return th, Vt[:4].T


def max_angle(A, B):
    sv = np.linalg.svd(A.T @ B, compute_uv=False)
    return float(np.degrees(np.arccos(min(1.0, sv.min()))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    U, W = rawpanel.universes(), rawpanel.wide()
    months = sorted(U[U.n_assets == P].formation_date.unique())
    rows = []
    t0 = time.time()
    for a, b in zip(months[:-1], months[1:]):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        sa = set(U[(U.n_assets == P) & (U.formation_date == a)].permno)
        sb = U[(U.n_assets == P) & (U.formation_date == b)].sort_values("rank").permno
        cols = np.array([c for c in sb if c in sa])
        tha, Ha = directions(W, a, cols)
        thb, Hb = directions(W, b, cols)
        gap = lambda th, j: float((th[j - 1] - th[j]) / th[j - 1])
        rows.append(dict(formation_date=b, common=len(cols),
                         **{f"ang{j}": float(np.degrees(np.arccos(min(1.0, abs(Ha[:, j - 1] @ Hb[:, j - 1])))))
                            for j in (1, 2, 3)},
                         span23=max_angle(Ha[:, 1:3], Hb[:, 1:3]), span123=max_angle(Ha[:, :3], Hb[:, :3]),
                         ang2_in23=max_angle(Ha[:, 1:3], Hb[:, 1:2]), ang3_in23=max_angle(Ha[:, 1:3], Hb[:, 2:3]),
                         ang2_in1234=max_angle(Ha, Hb[:, 1:2]),
                         gap12=gap(thb, 1), gap23=gap(thb, 2), mingap23=min(gap(tha, 2), gap(thb, 2))))
    pd.DataFrame(rows).to_csv(OUT / "separation.csv", index=False)
    print("done", round(time.time() - t0), "s,", len(rows), "pairs")


if __name__ == "__main__":
    main()

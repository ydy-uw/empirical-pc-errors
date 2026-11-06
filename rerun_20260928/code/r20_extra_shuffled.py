"""Extra understatement of A_j when the day variances are unrelated to the factor path.

The 63-day design of Section 6.2 (assumption3.run_month) plants the month's own day variances on their own
days, so the planted day variances keep the data's relation with the factor path, including the part that
estimating the leverage from the same panel adds (r18).  This repeats the design with a third arm whose
day-variance path is permuted over the days (a fresh permutation and calibration per panel), so that the
day variances move as much but are unrelated to the factor path.  Per month (clipped P = 3,000 panels,
63 days, k = the month's skewness count), with the calibration of assumption3.run_month (r12.planted):
  const   : 4 panels, constant day variance;
  own     : 4 panels, the month's day-variance path s_t^2 on its own days;
  shuffled: 4 panels, s_t^2 permuted over the days.
Per panel: und_j = arcsin sqrt(OE_j) - arcsin sqrt(A_j) in degrees, j <= min(k, 3).  Extra understatement:
per month the arm's mean und_j minus the constant arm's, then the median over months, as in
make_exhibits_0927.py.
Output (aggregates only): rerun_20260928/work/figure_code/out/checks_20261011/extra_shuffled_month.csv,
  extra_shuffled_summary.json
Run from the project root:  emp/.venv/Scripts/python.exe rerun_20260928/code/r20_extra_shuffled.py
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r12_null_checks as r12  # noqa: E402
import r15_leverage_check as r15  # noqa: E402

OUT = r15.OUT
REPS = 4


def deg(a):
    return np.degrees(np.arcsin(np.sqrt(np.clip(np.asarray(a, float), 0, 1))))


def one(args):
    mi, fd, k = args
    A3 = r12.A3
    Y = np.clip(r12._D["panels"][fd]["X"], -r12.CLIP, r12.CLIP)
    Y = Y - Y.mean(axis=0, keepdims=True)
    N = Y.shape[0]; m = min(k, 3)
    _, _, _, Vt, dvar, _ = A3.decompose(Y, k)
    Vk = Vt[:k].T
    rng = np.random.default_rng([20261012, mi, 4])
    ones = np.ones(N)
    fit0, s20, _ = r12.planted(Y, k, ones, rng)
    const = [A3.simulate(fit0, Vk, s20, ones, k, rng, False) for _ in range(REPS)]
    cv_data = A3.estimates(Y, k, False)["cv_dayvar"]
    cv_const = float(np.mean([c["cv_dayvar"] for c in const]))
    c = np.sqrt(max(0.0, 1.0 - cv_const ** 2 / cv_data ** 2))
    path = np.maximum(1.0 + c * (dvar / dvar.mean() - 1.0), 0.02); path /= path.mean()
    fit1, s21, _ = r12.planted(Y, k, path, rng)
    own = [A3.simulate(fit1, Vk, s21, path, k, rng, False) for _ in range(REPS)]
    shuf = []
    for _ in range(REPS):
        pth = path[rng.permutation(N)]
        fit2, s22, _ = r12.planted(Y, k, pth, rng)
        shuf.append(A3.simulate(fit2, Vk, s22, pth, k, rng, False))
    row = dict(formation_date=fd, k=k, shrink=c)
    for j in range(1, m + 1):
        u = {name: float(np.mean([deg(x[f"OE{j}"]) - deg(x[f"A{j}"]) for x in arm]))
             for name, arm in (("const", const), ("own", own), ("shuf", shuf))}
        row[f"und_const{j}"] = u["const"]
        row[f"extra_own{j}"] = u["own"] - u["const"]; row[f"extra_shuf{j}"] = u["shuf"] - u["const"]
    return row


def main():
    c = pd.read_csv(r12.COUNTS, parse_dates=["formation_date"])
    ks = c[c.n_assets == 3000].set_index("formation_date").k_skew
    t0 = time.time()
    with Pool(16, initializer=r12._init) as pool:
        rows = pool.map(one, [(i, pd.Timestamp(fd), int(ks.loc[fd])) for i, fd in enumerate(sorted(ks.index))],
                        chunksize=2)
    m = pd.DataFrame(rows); m.to_csv(OUT / "extra_shuffled_month.csv", index=False)
    summ = dict(months=len(m))
    for j in (1, 2, 3):
        g = m.dropna(subset=[f"extra_own{j}"])
        summ[f"dir{j}"] = dict(months=len(g), und_const=float(g[f"und_const{j}"].median()),
                               extra_own=float(g[f"extra_own{j}"].median()), extra_shuf=float(g[f"extra_shuf{j}"].median()),
                               own_pos=float((g[f"extra_own{j}"] > 0).mean()), shuf_pos=float((g[f"extra_shuf{j}"] > 0).mean()),
                               own_gt_shuf=float((g[f"extra_own{j}"] > g[f"extra_shuf{j}"]).mean()))
    (OUT / "extra_shuffled_summary.json").write_text(json.dumps(summ, indent=1))
    print(f"done in {time.time() - t0:.0f}s"); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

"""Rerun the finite-p accuracy simulation (figure_code/paper/p03_finite_p_error.py, unchanged code)
calibrated to the cleaned, clipped P = 3,000 panels.

p03 sets three inputs from the untreated panels of the uncleaned selection: the signal eigenvalues and noise level
(medians over the P = 3,000 panels with k >= 3), the coefficient of variation of the specific
variances (CV = 1.49), and the residual correlation in the block design (RHO = 0.077).  Here:
  theta_1..3, ell : medians over the cleaned, clipped P = 3,000 panels with skewness count >= 3
                    (01_fixed_clip/results/panels_c050.csv);
  CV              : split-sample CV of residual variances, cleaned clipped P = 3,000 (summary_c050.json);
  RHO             : mean residual correlation in the largest decile after 5 components, same source.
Everything else (designs, p grid, 400 replicates, seeds) is p03's.  As a self-check of the unchanged p03 code,
the same statistics are also computed from p03's output under its original calibration.
Output: rerun_20260928/work/figure_code/out/p03_20261006/{finite_p_error.csv, summary.json}
"""
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "rerun_20260928" / "work" / "figure_code" / "out" / "p03_20261006"
SCR = ROOT / "rerun_20260928" / "work" / "scratch_figs"
os.environ["HLHH_FIG_OUT"] = str(OUTD)
os.environ["HLHH_PAPER_FIGS"] = str(SCR); os.environ["HLHH_RECORD_FIGS"] = str(SCR)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(ROOT / "figure_code" / "paper"))
import numpy as np, pandas as pd  # noqa: E402
import p03_finite_p_error as p03  # noqa: E402

CLP = ROOT / "rerun_20260928" / "work" / "outlier_cleaning" / "experiments" / "01_fixed_clip" / "results"


def calibration_clean():
    d = pd.read_csv(CLP / "panels_c050.csv")
    g = d[(d.n_assets == 3000) & (d.k_skew >= 3)]
    th = np.array([g.theta1.median(), g.theta2.median(), g.theta3.median()])
    ell = float(g.ell.median())
    return th - ell, ell


def summarize(f):
    ok = f[f.p_eff >= 300]
    within2 = ((ok.gap_mean - ok.gap_mean_pred).abs() <= 2 * ok.gap_sd / np.sqrt(p03.REPS))
    return dict(
        sd_ratio={des: [float((g.gap_sd / g.gap_sd_pred).min()), float((g.gap_sd / g.gap_sd_pred).max())] for des, g in f.groupby("design")},
        sd_ratio_peff300={des: [float((g.gap_sd / g.gap_sd_pred).min()), float((g.gap_sd / g.gap_sd_pred).max())] for des, g in ok.groupby("design")},
        bias_within_2mcse_peff300=[int(within2.sum()), int(len(ok))],
        misses_peff300=[f"{r.design} p={int(r.p)} j={int(r.j)}" for r in ok[~within2].itertuples()],
        bias_negative_share=float((f.gap_mean < 0).mean()))


def main():
    s = json.load(open(CLP / "summary_c050.json"))
    p03.CV = float(s["accuracy"]["3000"]["cv_split"])
    p03.RHO = float(s["blocks"]["3000"]["m5"]) / 100.0
    p03.calibration = calibration_clean
    p03.OUT = OUTD
    OUTD.mkdir(parents=True, exist_ok=True)
    lam, ell = calibration_clean()
    print("calibration: lam", lam, "ell", ell, "CV", p03.CV, "RHO", p03.RHO, flush=True)
    p03.main()
    new = pd.read_csv(OUTD / "finite_p_error.csv")
    old = pd.read_csv(ROOT / "figure_code" / "out" / "finite_p_error.csv")
    out = dict(calibration=dict(lam=lam.tolist(), ell=ell, CV=p03.CV, RHO=p03.RHO),
               new=summarize(new), original=summarize(old))
    (OUTD / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

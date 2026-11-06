"""Eligible securities at every formation date, with SPAC flags and window volatilities.

For each of the 239 formation dates this runs the pipeline's own eligibility step
(emp/src/empirical_jse/crsp.py: build_eligible_universe, unchanged), and then

  1. checks that select_top_cap reproduces the frozen universes of the paper exactly
     (rolling_topcap_200601_202511_remote_v2_20260724) for P = 500, 1000, 2000, 3000;
  2. records, for every eligible security, its 63-day volatility, raw and winsorized
     (each stock's own 63 returns clipped at their 1st and 99th percentiles, numpy's default
     linear interpolation; annualized with sqrt(252), ddof = 1), and whether it is a SPAC on
     the formation date (spac_flags.py);
  3. stores the formation-date common-share rows needed to re-run the company-level
     representative choice after exclusions (choose_permco_representatives).

Output (licensed data, stays in emp/data):
  emp/data/processed/clean_20260928/candidates/year=YYYY/{eligible,common_rows}.parquet
  emp/data/processed/clean_20260928/candidates/year=YYYY/baseline_check.json
Run from emp/:  .venv/Scripts/python.exe ../rerun_20260928/code/r02_candidates.py 2006 2010
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EMP = ROOT / "emp"
sys.path.insert(0, str(EMP)); sys.path.insert(0, str(HERE))
from src.empirical_jse.crsp import (build_eligible_universe, select_top_cap,          # noqa: E402
                                    conventional_common_mask, choose_permco_representatives)
from src.empirical_jse.dataset import trading_calendar_from_reference, ParquetCIZCollection  # noqa: E402
import spac_flags  # noqa: E402

DATASET = EMP / "data/interim/crsp_ciz_parquet/crsp_ciz_2005_2025_remote_v1_20260723"
REFERENCE = EMP / "data/raw/wrds/crsp_ciz_reference/reference_1972_2025_remote_v1_20260723"
FROZEN = EMP / "data/processed/rolling/rolling_topcap_200601_202511_remote_v2_20260724"
OUT = EMP / "data/processed/clean_20260928/candidates"
SIZES = (500, 1000, 2000, 3000)
LOOKBACK = 63
ANN = np.sqrt(252.0)


def winsorized_vol(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Annualized volatility of each column, raw and after clipping at its own 1st/99th pct."""
    lo = np.percentile(R, 1, axis=0); hi = np.percentile(R, 99, axis=0)
    W = np.clip(R, lo, hi)
    return R.std(axis=0, ddof=1) * ANN, W.std(axis=0, ddof=1) * ANN


def main(y0: int, y1: int) -> None:
    cal = trading_calendar_from_reference(REFERENCE)
    coll = ParquetCIZCollection(DATASET)
    hist = spac_flags.security_history()
    frozen = pd.concat([pd.read_parquet(f, columns=["formation_date", "n_assets", "rank", "permno"])
                        for f in sorted(FROZEN.glob("year=*/selected_universes.parquet"))])
    frozen["formation_date"] = pd.to_datetime(frozen.formation_date)
    cache: dict[int, pd.DataFrame] = {}

    def year_frame(y):
        if y not in cache:
            for k in [k for k in cache if k < y - 1]:
                del cache[k]
            cache[y] = coll.load_years([y])
            cache[y]["dlycaldt"] = pd.to_datetime(cache[y]["dlycaldt"]).dt.normalize()
        return cache[y]

    for year in range(y0, y1 + 1):
        t0 = time.time()
        fds = sorted(frozen.loc[frozen.formation_date.dt.year == year, "formation_date"].unique())
        elig_rows, common_all, checks = [], [], []
        for fd in map(pd.Timestamp, fds):
            i = int(cal.searchsorted(fd)); assert cal[i] == fd
            est = cal[i - LOOKBACK + 1: i + 1]
            d = pd.concat([year_frame(y) for y in sorted({est[0].year, fd.year})], ignore_index=True)
            daily = d[(d.dlycaldt >= est[0]) & (d.dlycaldt <= fd)]
            el = build_eligible_universe(daily, cal, fd)
            assert tuple(el.window.estimation) == tuple(est)
            # 1. baseline reproduction
            for P in SIZES:
                mine = select_top_cap(el.representatives, P)
                ref = tuple(frozen[(frozen.formation_date == fd) & (frozen.n_assets == P)]
                            .sort_values("rank").permno.astype(int))
                checks.append(dict(formation_date=str(fd.date()), n_assets=P, identical=mine == ref))
                if mine != ref:
                    raise RuntimeError(f"baseline differs from the frozen universe at {fd.date()} P={P}")
            # 3. common rows at formation, and a check that the representative step is reproducible
            frm = daily[daily.dlycaldt == fd]
            common = frm.loc[conventional_common_mask(frm), ["permno", "permco", "dlycap", "dlycapflg"]].copy()
            rep2 = choose_permco_representatives(common, el.eligible_permnos)
            assert rep2.permno.tolist() == el.representatives.permno.tolist()
            common.insert(0, "formation_date", fd); common_all.append(common)
            # 2. volatilities and SPAC flags of every eligible security
            elig = list(el.eligible_permnos)
            R = (daily[daily.permno.isin(elig)].pivot(index="dlycaldt", columns="permno", values="dlyret")
                 .reindex(index=est, columns=elig).to_numpy(float))
            assert not np.isnan(R).any()
            vol, volw = winsorized_vol(R)
            spac = spac_flags.spac_permnos_at(hist, fd)
            reps = el.representatives.reset_index(drop=True)
            rank = pd.Series(np.arange(1, len(reps) + 1), index=reps.permno.astype(int))
            e = pd.DataFrame(dict(formation_date=fd, permno=np.array(elig, dtype=np.int64), vol=vol, vol_w=volw))
            e["spac"] = e.permno.isin(spac)
            e["baseline_rank"] = e.permno.map(rank)          # rank among company representatives
            e["min_ret"] = R.min(axis=0); e["max_ret"] = R.max(axis=0)
            elig_rows.append(e)
        od = OUT / f"year={year}"; od.mkdir(parents=True, exist_ok=True)
        pd.concat(elig_rows).to_parquet(od / "eligible.parquet", index=False)
        pd.concat(common_all).to_parquet(od / "common_rows.parquet", index=False)
        (od / "baseline_check.json").write_text(json.dumps(checks, indent=0))
        print(f"{year}: {len(fds)} months, baseline identical in {sum(c['identical'] for c in checks)}"
              f"/{len(checks)} cells, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))

"""Integrity checks of the cleaned universes (28 September 2026 rerun).  Exits non-zero on failure."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EMP = HERE.parents[1] / "emp"
sys.path.insert(0, str(HERE))
import spac_flags  # noqa: E402

NEW = EMP / "data/processed/rolling/rolling_topcap_clean_200601_202511_20260928"
OLD = EMP / "data/processed/rolling/rolling_topcap_200601_202511_remote_v2_20260724"
CAND = EMP / "data/processed/clean_20260928/candidates"
rd = lambda root, cols=None: pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(root.glob("year=*/selected_universes.parquet"))])
new = rd(NEW); old = rd(OLD, ["formation_date", "n_assets", "rank", "permno", "permco"])
el = pd.concat([pd.read_parquet(f) for f in sorted(CAND.glob("year=*/eligible.parquet"))])
ex = pd.read_parquet(NEW / "exclusions.parquet")
fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name); (None if ok else fails.append(name))

check("239 formation dates, same as the frozen run", sorted(new.formation_date.unique()) == sorted(pd.to_datetime(old.formation_date).unique()) and new.formation_date.nunique() == 239)
g = new.groupby(["formation_date", "n_assets"])
check("every cell has exactly P rows with ranks 1..P", all((len(v) == P and (v["rank"].to_numpy() == np.arange(1, P + 1)).all()) for (fd, P), v in g))
check("no duplicate permno or permco within a cell", not new.duplicated(["formation_date", "n_assets", "permno"]).any() and not new.duplicated(["formation_date", "n_assets", "permco"]).any())
nest = True
for fd, v in new.groupby("formation_date"):
    s = {P: set(v[v.n_assets == P].permno) for P in (500, 1000, 2000, 3000)}
    nest &= s[500] <= s[1000] <= s[2000] <= s[3000]
    top = v[v.n_assets == 3000].sort_values("rank")
    for P in (500, 1000, 2000):
        nest &= list(top.permno[:P]) == list(v[v.n_assets == P].sort_values("rank").permno)
check("universes nested, and top-P is the first P of top-3000", nest)
m = new.merge(el[["formation_date", "permno", "spac", "vol_w"]], on=["formation_date", "permno"], how="left", suffixes=("", "_el"))
check("every selected security is eligible", m.spac.notna().all())
check("no SPAC selected", not m.spac.astype(bool).any())
check("no selected security has vol_w < 8%", (m.vol_w_el >= 0.08).all())
check("company cap non-increasing in rank", all(np.all(np.diff(v.sort_values("rank").company_cap.to_numpy()) <= 0) for _, v in g))
check("kept + SPAC removed + low-vol removed = P in every cell", ((ex.kept_from_original + ex.spac_removed + ex.lowvol_removed) == ex.n_assets).all())
# independent recomputation of the SPAC flag for a sample of months
h = spac_flags.security_history()
ok = True
for fd in sorted(new.formation_date.unique())[::20]:
    s = spac_flags.spac_permnos_at(h, fd)
    ok &= not (set(new[(new.formation_date == fd)].permno) & s)
check("SPAC flags recomputed independently on 12 months: none selected", ok)
# relation to the original universes
o3 = old[old.n_assets == 3000]; n3 = new[new.n_assets == 3000]
j = o3.merge(n3, on=["formation_date", "permno"], how="outer", indicator=True)
print("P=3000 stock-months: original", len(o3), " cleaned", len(n3), " in both", (j._merge == "both").sum(),
      " only original", (j._merge == "left_only").sum(), " only cleaned", (j._merge == "right_only").sum())
sys.exit(1 if fails else 0)

"""Run one outlier treatment on all 956 formation panels and write its experiment folder.

    python outlier_cleaning/code/run_method.py 01_fixed_clip            # all settings + heavy parts
    python outlier_cleaning/code/run_method.py 01_fixed_clip --core-only

Writes, under outlier_cleaning/experiments/<method>/:
    config.json                       the treatment, its settings, code hashes, run times
    results/panels_<setting>.csv      one row per panel (956 rows): what the treatment
                                      changed and every per-panel diagnostic
    results/blocks.csv                residual size blocks, main setting, P = 500 and 3,000
    results/fa.csv                    factor-analytic estimate, main setting, all panels
    results/oos.csv                   disjoint-window check, main setting, P = 500 and 3,000
    logs/run.log, logs/input_check_before.json, logs/input_check_after.json
All outputs are aggregates; no security identifiers are written.
The reference count k_ref is the skewness count of the untreated panel, read from
experiments/00_raw/results/panels_main.csv (so 00_raw runs first).
"""
from __future__ import annotations
import sys, time, argparse, platform
import numpy as np, pandas as pd
from oc_common import EXP, HERE, P_GRID, N_OBS, sha256, dump
import panels, treatments, diagnostics, heavy, integrity


def log(msg, fh):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); fh.write(line + "\n"); fh.flush()


def k_reference():
    f = EXP / "00_raw" / "results" / "panels_main.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f, parse_dates=["formation_date"])
    return d.set_index(["formation_date", "n_assets"]).k_skew


def run_core(method, setting, kref, fh):
    rows = []
    for P in P_GRID:
        d = panels.load(P)
        for fd in sorted(d):
            pan = d[fd]
            Xt, keep, info = treatments.apply(method, setting, pan["X"], pan)
            kr = int(kref.loc[(fd, P)]) if kref is not None else None
            row = dict(formation_date=fd, n_assets=P, setting=setting, **info)
            row.update(diagnostics.core(Xt, kr))
            rows.append(row)
        log(f"  {setting}: P={P} done", fh)
    return pd.DataFrame(rows)


def run_heavy(method, core, fh, setting="main"):
    kown = core.set_index(["formation_date", "n_assets"]).k_skew
    brows, frows, orows = [], [], []
    for P in P_GRID:
        d = panels.load(P)
        months = sorted(d)
        for fd in months:
            pan = d[fd]
            Xt, keep, _ = treatments.apply(method, setting, pan["X"], pan)
            k = int(kown.loc[(fd, P)])
            frows.append(dict(formation_date=fd, n_assets=P, **heavy.fa(Xt, k)))
            if P in (500, 3000):
                for b in heavy.blocks(Xt, k):
                    brows.append(dict(formation_date=fd, n_assets=P, k=k, **b))
        log(f"  heavy: blocks and factor-analytic fits, P={P} done", fh)
        if P not in (500, 3000):
            continue
        for i, t in enumerate(months):
            d1 = d[t]["dates"]; t2 = None
            for L in (3, 4):
                if i + L < len(months) and d[months[i + L]]["dates"][0] > d1[-1]:
                    t2 = months[i + L]; break
            if t2 is None:
                continue
            a, b = d[t], d[t2]
            names, ia, ib = np.intersect1d(a["permnos"], b["permnos"], return_indices=True)
            ctx1 = {k_: v[ia] for k_, v in a.items() if k_ in ("prior_scale", "prior_ret")}
            ctx2 = {k_: v[ib] for k_, v in b.items() if k_ in ("prior_scale", "prior_ret")}
            X1, keep1, _ = treatments.apply(method, setting, a["X"][:, ia], ctx1)
            X2, keep2, _ = treatments.apply(method, setting, b["X"][:, ib], ctx2)
            if keep1 is not None:                 # a name dropped in either window leaves both
                both = keep1 & keep2
                X1 = a["X"][:, ia][:, both]; X2 = b["X"][:, ib][:, both]
                X1, _, _ = treatments.apply(method, setting, X1, None)
                X2, _, _ = treatments.apply(method, setting, X2, None)
            k1, k2 = int(kown.loc[(t, P)]), int(kown.loc[(t2, P)])
            orows.append(dict(t=t, t2=t2, n_assets=P, names=X1.shape[1],
                              **heavy.oos_pair(X1, X2, k1, k2)))
        log(f"  heavy: disjoint windows, P={P} done", fh)
    return pd.DataFrame(brows), pd.DataFrame(frows), pd.DataFrame(orows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=sorted(treatments.METHODS))
    ap.add_argument("--core-only", action="store_true")
    ap.add_argument("--settings", nargs="*")
    ap.add_argument("--heavy-setting", help="run only the slower diagnostics, for this non-main setting")
    a = ap.parse_args()
    out = EXP / a.method; (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    fh = open(out / "logs" / "run.log", "a")
    t0 = time.time()
    log(f"start {a.method}", fh)
    before = integrity.check(); dump(before, out / "logs" / "input_check_before.json")
    log(f"input check before: {before['files']} files, manifest hashes match: {before['all_match']}", fh)
    kref = None if a.method == "00_raw" else k_reference()
    if a.method != "00_raw" and kref is None:
        raise SystemExit("run 00_raw first: its skewness counts are the reference count")
    fname, settings = treatments.METHODS[a.method]
    todo = [] if a.heavy_setting else (a.settings or list(settings))
    timings = {}
    for s in todo:
        t1 = time.time()
        core = run_core(a.method, s, kref, fh)
        if a.method == "00_raw":
            from p01_panel_stats import jung_counts
            kp = jung_counts()
            core["k_pipeline"] = [int(kp.loc[(f, P)]) for f, P in zip(core.formation_date, core.n_assets)]
        core.to_csv(out / "results" / f"panels_{s}.csv", index=False)
        timings[s] = round(time.time() - t1, 1)
        log(f"setting {s}: {len(core)} panels, {timings[s]}s", fh)
    if not a.core_only and "main" in todo:
        t1 = time.time()
        core = pd.read_csv(out / "results" / "panels_main.csv", parse_dates=["formation_date"])
        b, f, o = run_heavy(a.method, core, fh)
        b.to_csv(out / "results" / "blocks.csv", index=False)
        f.to_csv(out / "results" / "fa.csv", index=False)
        o.to_csv(out / "results" / "oos.csv", index=False)
        timings["heavy"] = round(time.time() - t1, 1)
        log(f"heavy parts: {timings['heavy']}s", fh)
    if a.heavy_setting:
        hs = a.heavy_setting; t1 = time.time()
        core = pd.read_csv(out / "results" / f"panels_{hs}.csv", parse_dates=["formation_date"])
        b, f, o = run_heavy(a.method, core, fh, hs)
        b.to_csv(out / "results" / f"blocks_{hs}.csv", index=False)
        f.to_csv(out / "results" / f"fa_{hs}.csv", index=False)
        o.to_csv(out / "results" / f"oos_{hs}.csv", index=False)
        timings[f"heavy_{hs}"] = round(time.time() - t1, 1)
        log(f"heavy parts for setting {hs}: {timings[f'heavy_{hs}']}s", fh)
    after = integrity.check(); dump(after, out / "logs" / "input_check_after.json")
    same = integrity.unchanged(before, after)
    log(f"input check after: manifest hashes match: {after['all_match']}; "
        f"every input unchanged by the run: {same}", fh)
    code_files = ["oc_common.py", "panels.py", "treatments.py", "diagnostics.py", "heavy.py",
                  "run_method.py", "integrity.py"]
    if a.heavy_setting and (out / "config.json").is_file():
        import json
        cfg = json.load(open(out / "config.json"))
        cfg.setdefault("runtime_seconds", {}).update(timings)
        cfg.setdefault("heavy_settings_run", []).append(a.heavy_setting)
        cfg["inputs_unchanged"] = bool(cfg.get("inputs_unchanged", True) and same)
        cfg["inputs_match_manifests"] = bool(cfg.get("inputs_match_manifests", True) and after["all_match"])
        dump(cfg, out / "config.json")
        log(f"done {a.method} (setting {a.heavy_setting}) in {time.time()-t0:.0f}s", fh)
        if not same:
            raise SystemExit("INPUT FILES CHANGED DURING THE RUN")
        return
    dump(dict(method=a.method, treatment=fname, settings=settings, main_setting=settings["main"],
              settings_run=todo, core_only=a.core_only, inputs_unchanged=same,
              inputs_match_manifests=after["all_match"],
              python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
              code_sha256={c: sha256(HERE / c) for c in code_files},
              runtime_seconds=timings, finished=time.strftime("%Y-%m-%d %H:%M:%S")),
         out / "config.json")
    log(f"done {a.method} in {time.time()-t0:.0f}s", fh)
    if not same:
        raise SystemExit("INPUT FILES CHANGED DURING THE RUN")


if __name__ == "__main__":
    main()

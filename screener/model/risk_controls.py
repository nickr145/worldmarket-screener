"""Sensitivities on the risk targets (confirmatory set, as-of 2022-2024, country-grouped CV, TR excluded):
 (1) composition control: industry mean + n_firms + top-firm concentration (per-industry ridge) vs trailing persistence;
 (2) persistence on the USD-converted return series.
python -m screener.model.risk_controls -> reports/risk_controls.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from screener.features.run_panel import ROOT, vp
from screener.model.baselines import B, SEED, metrics, ridge_fit_predict
from screener.model.risk_models import TARGETS, dataset

warnings.filterwarnings("ignore")
COMP = ["n_members", "top1_share"]


def with_panel(d):
    p = pd.read_parquet(vp("data/panel/panel.parquet"))[["country", "industry", "as_of_year", "n_members", "top1_share"]]
    return d.merge(p, on=["country", "industry", "as_of_year"], how="left")


def oof(p, trail):
    pr = {"trailing_persistence": p[trail].values, "industry_mean": np.full(len(p), np.nan), "composition_ridge": np.full(len(p), np.nan)}
    for tr, te in GroupKFold(n_splits=10).split(p, groups=p.country):
        trn, tst = p.iloc[tr], p.iloc[te]
        im = tst.industry.map(trn.groupby("industry").y.mean()).fillna(trn.y.mean()).values
        pr["industry_mean"][te] = im
        c = im.copy()
        for ind in tst.industry.unique():
            m = (tst.industry == ind).values
            r = ridge_fit_predict(trn[trn.industry == ind], tst[m], COMP)
            if r is not None:
                c[m] = r
        pr["composition_ridge"][te] = c
    return pr


def evaluate(p, trail, label, target, rows, rng):
    y, country = p.y.values, p.country.values
    grp = pd.factorize(p.industry + p.as_of_year.astype(str))[0]
    pr = oof(p, trail)
    cs = np.unique(country)
    idx = {c: np.where(country == c)[0] for c in cs}
    draws = [np.concatenate([idx[c] for c in rng.choice(cs, len(cs), replace=True)]) for _ in range(B)]
    boot = {k: pd.DataFrame([metrics(y[x], v[x], grp[x]) for x in draws]) for k, v in pr.items()}
    est = {k: metrics(y, v, grp) for k, v in pr.items()}
    for k in pr:
        e, lo, hi = dict(est[k]), boot[k].quantile(0.025), boot[k].quantile(0.975)
        if k in ("industry_mean",):
            for kk in ("spearman", "spread"):
                e[kk], lo[kk], hi[kk] = np.nan, np.nan, np.nan
        rows.append({"sensitivity": label, "target": target, "kind": "metrics", "model": k, "n_rows": len(p), **e,
                     **{f"{m}_lo": lo[m] for m in e}, **{f"{m}_hi": hi[m] for m in e}})
    for base in ("composition_ridge", "industry_mean"):
        for metric, sign in (("rmse", -1), ("spearman", 1)):
            if base == "industry_mean" and metric == "spearman":
                continue
            diff = sign * (boot["trailing_persistence"][metric] - boot[base][metric])  # positive = persistence better
            rows.append({"sensitivity": label, "target": target, "kind": "paired persistence-minus-" + base, "model": metric,
                         "n_rows": len(p), "diff_persistence_better": sign * (est["trailing_persistence"][metric] - est[base][metric]),
                         "diff_lo": diff.quantile(0.025), "diff_hi": diff.quantile(0.975)})


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for usd, label in ((False, "composition control (local-currency series)"), (True, "USD-converted series")):
        d = dataset() if not usd else None
        if usd:  # same dataset construction, USD price targets
            pt = pd.read_parquet(vp("data/panel/price_targets_usd.parquet"))
            base = dataset().drop(columns=[c for c in pt.columns if c not in ("country", "industry", "as_of_year")], errors="ignore")
            d = base.merge(pt, on=["country", "industry", "as_of_year"])
        d = with_panel(d)
        d = d[~d.membership_lookahead & d.as_of_year.between(2022, 2024)]
        for target, trail in TARGETS.items():
            p = d[d[target].notna() & d[trail].notna() & (d[COMP].notna().all(axis=1) if not usd else True)].assign(y=lambda x: x[target]).reset_index(drop=True)
            evaluate(p, trail, label, target, rows, rng)
    r = pd.DataFrame(rows)
    r.to_csv(vp("reports/risk_controls.csv"), index=False)
    return r


if __name__ == "__main__":
    r = main()
    f = lambda x, m: "n/a" if pd.isna(x.get(m)) else f"{x[m]:.3f} [{x[m + '_lo']:.3f},{x[m + '_hi']:.3f}]"
    for (lab, tg), g in r.groupby(["sensitivity", "target"], sort=False):
        m = g[g.kind == "metrics"]
        print(f"\n=== {lab} | {tg} | n={m.n_rows.iloc[0]} ===")
        print(pd.DataFrame({"model": m.model, "RMSE": m.apply(lambda x: f(x, "rmse"), axis=1), "R2": m.apply(lambda x: f(x, "r2"), axis=1),
                            "Spearman": m.apply(lambda x: f(x, "spearman"), axis=1), "spread": m.apply(lambda x: f(x, "spread"), axis=1)}).to_string(index=False))
        for _, x in g[g.kind != "metrics"].iterrows():
            print(f"  {x.kind} on {x.model}: {x.diff_persistence_better:+.4f} [{x.diff_lo:+.4f},{x.diff_hi:+.4f}]  (positive = persistence better)")

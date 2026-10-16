"""Baselines + boosted models on the price-based risk targets, per the registered risk-target protocol (PLAN.md).

python -m screener.model.risk_models -> reports/risk_targets.csv, reports/risk_promotion_tests.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from screener.features.build_panel import decision_date, macro_asof
from screener.features.run_panel import ROOT, vp, load_cfg, load_macro
from screener.model.baselines import B, CONSTANT, MIN_TRAIN, SEED, SHARED, SPECIFIC, metrics, ridge_fit_predict
from screener.model.boosted import GRID, MODELS
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")
TARGETS = {"dd_dev_12m": "trail_dd_dev_12m", "mdd_12m": "trail_mdd_12m"}


def dataset():
    st, countries, _ = load_cfg()
    macro = load_macro()
    macro = macro[macro.iso3.isin([m["iso3"] for m in countries.values()])]
    pt = pd.read_parquet(vp("data/panel/price_targets.parquet"))
    fu = pd.read_csv(ROOT / "reports/firm_universe.csv")
    n_listed = fu[fu.domestic].groupby(["country", "industry"]).size().rename("n_listed")
    rows = []
    for r in pt.itertuples():
        m, _ = macro_asof(macro, countries[r.country]["iso3"], decision_date(r.as_of_year, st))
        d = r._asdict()
        d.update(m)
        d["log_gdp_pc"] = np.log(m.get("gdp_pc_usd", np.nan))
        d["firms_per_gdp"] = n_listed.get((r.country, r.industry), np.nan) / (m.get("gdp_usd", np.nan) / 1e9)
        d["region"] = countries[r.country]["region"]
        rows.append(d)
    df = pd.DataFrame(rows)
    return df[~df.country.isin(st["exclude_from_training"])].reset_index(drop=True)


def predict(tr, te, trail, boosted=True):
    out = {"global_mean": np.full(len(te), tr.y.mean())}
    out["industry_mean"] = te.industry.map(tr.groupby("industry").y.mean()).fillna(tr.y.mean()).values
    out["trailing_persistence"] = te[trail].values
    for name, extra in (("ridge_macro", []), ("ridge_trailing_measure", [trail])):
        pred = out["industry_mean"].copy()
        for ind in te.industry.unique():
            m = (te.industry == ind).values
            r = ridge_fit_predict(tr[tr.industry == ind], te[m], SPECIFIC[ind] + SHARED + extra)
            if r is not None:
                pred[m] = r
        out[name] = pred
    if boosted:
        for name in MODELS:
            pred = out["industry_mean"].copy()
            for ind in te.industry.unique():
                cols = SPECIFIC[ind] + SHARED + [trail]
                t = tr[tr.industry == ind]
                m = (te.industry == ind).values
                if len(t) < MIN_TRAIN or t.country.nunique() < 5:
                    continue
                gs = GridSearchCV(MODELS[name](), GRID, scoring="neg_mean_squared_error", cv=GroupKFold(n_splits=min(5, t.country.nunique())))
                gs.fit(t[cols], t.y, groups=t.country)
                pred[m] = gs.predict(te.loc[m, cols])
            out[name] = pred
    return out


def folds(p, scheme):
    if scheme == "country":
        yield from GroupKFold(n_splits=10).split(p, groups=p.country)
    else:
        for r in p.region.unique():
            yield np.where(p.region != r)[0], np.where(p.region == r)[0]


def run(d, target, label):
    trail = TARGETS[target]
    p = d[d[target].notna() & d[trail].notna()].assign(y=lambda x: x[target]).reset_index(drop=True)
    y, country = p.y.values, p.country.values
    grp = pd.factorize(p.industry + p.as_of_year.astype(str))[0]
    rng = np.random.default_rng(SEED)
    rows, tests = [], []
    for scheme in ("country", "region"):
        preds = {}
        for tr, te in folds(p, scheme):
            for k, v in predict(p.iloc[tr], p.iloc[te], trail).items():
                preds.setdefault(k, np.full(len(p), np.nan))[te] = v
        cs = np.unique(country)
        idx = {c: np.where(country == c)[0] for c in cs}
        draws = [np.concatenate([idx[c] for c in rng.choice(cs, len(cs), replace=True)]) for _ in range(B)]
        boot = {k: pd.DataFrame([metrics(y[x], pr[x], grp[x]) for x in draws]) for k, pr in preds.items()}
        est_all = {k: metrics(y, pr, grp) for k, pr in preds.items()}
        for k in preds:
            est, lo, hi = dict(est_all[k]), boot[k].quantile(0.025), boot[k].quantile(0.975)
            for kk in ("spearman", "spread"):
                if k in CONSTANT:
                    est[kk], lo[kk], hi[kk] = np.nan, np.nan, np.nan
            rows.append({"set": label, "target": target, "cv": scheme, "model": k, "n_rows": len(p), "n_countries": len(cs), **est,
                         **{f"{m}_lo": lo[m] for m in est}, **{f"{m}_hi": hi[m] for m in est}})
        bases = ["trailing_persistence", "industry_mean", "ridge_macro", "ridge_trailing_measure"]
        best = min(bases, key=lambda k: est_all[k]["rmse"])
        for k in MODELS:
            for base, metric, sign in (("ridge_trailing_measure", "rmse", -1), ("trailing_persistence", "spearman", 1), (best, "rmse", -1)):
                diff = sign * (boot[k][metric] - boot[base][metric])
                est = sign * (est_all[k][metric] - est_all[base][metric])
                lo, hi = diff.quantile(0.025), diff.quantile(0.975)
                tests.append({"set": label, "target": target, "cv": scheme, "model": k, "vs": base, "best_rmse_baseline": best,
                              "metric": metric, "diff_model_better": est, "lo": lo, "hi": hi, "favours_model_significantly": bool(lo > 0)})
    return rows, tests


def main():
    d = dataset()
    sets = {"confirmatory 2022-2024": d[~d.membership_lookahead & d.as_of_year.between(2022, 2024)],
            "exploratory 2016-2021 (membership_lookahead)": d[d.membership_lookahead]}
    R, T = [], []
    for label, dd in sets.items():
        for target in TARGETS:
            r, t = run(dd, target, label)
            R += r
            T += t
            print("done", label, target, flush=True)
    r, t = pd.DataFrame(R), pd.DataFrame(T)
    r.to_csv(vp("reports/risk_targets.csv"), index=False)
    t.to_csv(vp("reports/risk_promotion_tests.csv"), index=False)
    return r, t


if __name__ == "__main__":
    main()

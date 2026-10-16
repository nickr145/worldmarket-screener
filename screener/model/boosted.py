"""XGBoost / LightGBM run once, as pre-registered (PLAN.md): primary spec, per-industry models, 8 fixed features, nested tuning.

python -m screener.model.boosted -> reports/boosted.csv, reports/promotion_tests.csv
"""
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from xgboost import XGBRegressor

from screener.features.run_panel import ROOT, vp
from screener.model.baselines import B, CONSTANT, MIN_TRAIN, SEED, SHARED, SPECIFIC, folds, load, metrics, predict as base_predict

warnings.filterwarnings("ignore")
PRIMARY = "PRIMARY h1 real | trailing | no-TR | no-biotech"
GRID = {"max_depth": [2, 3], "n_estimators": [50, 150], "reg_lambda": [5, 20]}
FIXED = dict(learning_rate=0.05, subsample=0.8, random_state=0)
MODELS = {
    "xgboost": lambda: XGBRegressor(**FIXED, colsample_bytree=0.8, min_child_weight=5, n_jobs=1, verbosity=0),
    "lightgbm": lambda: LGBMRegressor(**FIXED, subsample_freq=1, colsample_bytree=0.8, min_child_samples=5, num_leaves=7,
                                      n_jobs=1, verbose=-1),
}


def boosted_predict(tr, te, name, base_fallback):
    pred = base_fallback.copy()
    for ind in te.industry.unique():
        cols = SPECIFIC[ind] + SHARED + ["trailing_growth"]
        t = tr[tr.industry == ind]
        m = (te.industry == ind).values
        if len(t) < MIN_TRAIN or t.country.nunique() < 5:
            continue
        gs = GridSearchCV(MODELS[name](), GRID, scoring="neg_mean_squared_error",
                          cv=GroupKFold(n_splits=min(5, t.country.nunique())))  # nested: inside the training fold only
        gs.fit(t[cols], t.y, groups=t.country)
        pred[m] = gs.predict(te.loc[m, cols])
    return pred


def oof(p, scheme):
    preds = {}
    for tr, te in folds(p, scheme):
        trn, tst = p.iloc[tr], p.iloc[te]
        out = base_predict(trn, tst, True)
        for name in MODELS:
            out[name] = boosted_predict(trn, tst, name, out["industry_mean"])
        for k, v in out.items():
            preds.setdefault(k, np.full(len(p), np.nan))[te] = v
    return preds


def main():
    p, _ = load(PRIMARY)
    y, country = p.y.values, p.country.values
    grp = pd.factorize(p.industry + p.as_of_year.astype(str))[0]
    rng = np.random.default_rng(SEED)
    rows, tests = [], []
    for scheme in ("country", "region"):
        preds = oof(p, scheme)
        cs = np.unique(country)
        idx = {c: np.where(country == c)[0] for c in cs}
        draws = [np.concatenate([idx[c] for c in rng.choice(cs, len(cs), replace=True)]) for _ in range(B)]  # shared by all models
        boot = {k: pd.DataFrame([metrics(y[d], pr[d], grp[d]) for d in draws]) for k, pr in preds.items()}
        for k, pr in preds.items():
            est = metrics(y, pr, grp)
            lo, hi = boot[k].quantile(0.025), boot[k].quantile(0.975)
            for kk in ("spearman", "spread"):
                if k in CONSTANT:
                    est[kk], lo[kk], hi[kk] = np.nan, np.nan, np.nan
            rows.append({"cv": scheme, "model": k, "n_rows": len(p), "n_countries": len(cs), **est,
                         **{f"{m}_lo": lo[m] for m in est}, **{f"{m}_hi": hi[m] for m in est}})
        for k in MODELS:
            for base, metric, sign in (("ridge_full", "rmse", -1), ("trailing_persistence", "spearman", 1)):
                diff = sign * (boot[k][metric] - boot[base][metric])  # positive = boosted model better
                est = sign * (metrics(y, preds[k], grp)[metric] - metrics(y, preds[base], grp)[metric])
                lo, hi = diff.quantile(0.025), diff.quantile(0.975)
                tests.append({"cv": scheme, "model": k, "vs": base, "metric": metric, "diff_model_better": est, "lo": lo, "hi": hi,
                              "favours_model_significantly": bool(lo > 0)})
    r, t = pd.DataFrame(rows), pd.DataFrame(tests)
    r.to_csv(vp("reports/boosted.csv"), index=False)
    t.to_csv(vp("reports/promotion_tests.csv"), index=False)
    return r, t


if __name__ == "__main__":
    r, t = main()
    f = lambda x, m: "n/a" if pd.isna(x[m]) else f"{x[m]:.3f} [{x[m + '_lo']:.3f},{x[m + '_hi']:.3f}]"
    pd.set_option("display.width", 250)
    for cv in ("country", "region"):
        d = r[r.cv == cv].copy()
        for m in ("rmse", "r2", "spearman", "spread"):
            d[m] = d.apply(lambda x: f(x, m), axis=1)
        print(f"\n=== PRIMARY | CV: {cv} | n={d.n_rows.iloc[0]}, {d.n_countries.iloc[0]} countries ===")
        print(d[["model", "rmse", "r2", "spearman", "spread"]].to_string(index=False))
    print("\n=== Paired country-clustered bootstrap of the metric DIFFERENCE (positive = boosted model better) ===")
    t["diff [95% CI]"] = t.apply(lambda x: f"{x.diff_model_better:+.3f} [{x.lo:+.3f},{x.hi:+.3f}]", axis=1)
    print(t[["cv", "model", "vs", "metric", "diff [95% CI]", "favours_model_significantly"]].to_string(index=False))
    c = t[t.cv == "country"]
    a = c[(c.vs == "ridge_full")].set_index("model").favours_model_significantly
    b = c[(c.vs == "trailing_persistence")].set_index("model").favours_model_significantly
    print("\nPROMOTION (country-grouped CV; needs BOTH (a) and (b)):", {m: bool(a[m] and b[m]) for m in MODELS})

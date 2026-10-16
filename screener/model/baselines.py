"""Baselines under the pre-registered protocol (PLAN.md, 'Evaluation protocol'). No boosted trees here.

python -m screener.model.baselines  -> reports/baselines.csv
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from screener.features.run_panel import ROOT, vp

warnings.filterwarnings("ignore")
SPECIFIC = {
    "non_bank_financials": ["account_ownership_pct", "credit_private_pct_gdp", "bank_npl_pct", "mobile_subs_per_100", "wgi_reg_quality"],
    "software": ["internet_users_pct", "secure_servers_per_million", "ict_service_exports_pct", "rd_pct_gdp", "wgi_reg_quality"],
    "retail": ["internet_users_pct", "urban_pop_pct", "household_consumption_pct_gdp", "mobile_subs_per_100", "lpi_overall"],
    "healthcare": ["health_exp_pct_gdp", "oop_share_pct", "pop_65plus_pct", "physicians_per_1000", "life_expectancy"],
}
SHARED = ["log_gdp_pc", "firms_per_gdp"]  # + trailing_growth in the "full" list
B, SEED, MIN_TRAIN = 1000, 0, 15
CONSTANT = {"global_mean", "industry_mean"}
ALPHAS = np.logspace(-1, 3, 9)

SPECS = {  # name: (panel suffix, target, need trailing, row filter)
    "PRIMARY h1 real | trailing | no-TR | no-biotech": ("", "target_h1", True, lambda d: d[d.train_eligible]),
    "S1 h2 real": ("", "target_h2", True, lambda d: d[d.train_eligible]),
    "S2 no-trailing variant": ("", "target_h1", False, lambda d: d[d.train_eligible]),
    "S3 USD target": ("", "target_h1_usd", True, lambda d: d[d.train_eligible]),
    "S4 TR included": ("", "target_h1", True, lambda d: d),
    "S5 biotech included": ("_biotech", "target_h1", True, lambda d: d[d.train_eligible]),
    "S6 concentrated cells excluded": ("", "target_h1", True, lambda d: d[d.train_eligible & ~d.concentrated]),
    "S7 domicile_assumed (KR) firms excluded": ("_no_assumed", "target_h1", True, lambda d: d[d.train_eligible]),
}


def load(spec):
    suf, y, trail, filt = SPECS[spec]
    p = pd.read_parquet(vp(f"data/panel/panel{suf}.parquet"))
    p = p[(p.status == "ok") & p[y].notna()]
    p = filt(p)
    if trail:
        p = p[p.trailing_growth.notna()]
    p = p.assign(log_gdp_pc=np.log(p.gdp_pc_usd), y=p[y]).reset_index(drop=True)
    return p, trail


def ridge_fit_predict(tr, te, cols):
    if len(tr) < MIN_TRAIN or tr.country.nunique() < 5:
        return None
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
    gs = GridSearchCV(pipe, {"ridge__alpha": ALPHAS}, scoring="neg_mean_squared_error",
                      cv=GroupKFold(n_splits=min(5, tr.country.nunique())))
    gs.fit(tr[cols], tr.y, groups=tr.country)
    return gs.predict(te[cols])


def predict(tr, te, trail):
    out = {"global_mean": np.full(len(te), tr.y.mean())}
    im = tr.groupby("industry").y.mean()
    out["industry_mean"] = te.industry.map(im).fillna(tr.y.mean()).values
    if trail:
        out["trailing_persistence"] = te.trailing_growth.values
    for name, extra in [("ridge_macro", []), ("ridge_full", ["trailing_growth"])]:
        if name == "ridge_full" and not trail:
            continue
        pred = out["industry_mean"].copy()
        for ind in te.industry.unique():
            cols = SPECIFIC[ind] + SHARED + extra
            m = (te.industry == ind).values
            r = ridge_fit_predict(tr[tr.industry == ind], te[m], cols)
            if r is not None:
                pred[m] = r
        out[name] = pred
    return out


def folds(p, scheme):
    if scheme == "country":
        for tr, te in GroupKFold(n_splits=10).split(p, groups=p.country):
            yield tr, te
    elif scheme == "region":
        for r in p.region.unique():
            yield np.where(p.region != r)[0], np.where(p.region == r)[0]
    elif scheme == "time":  # exploratory: train as-of 2023 -> test as-of 2024
        yield np.where(p.as_of_year == 2023)[0], np.where(p.as_of_year == 2024)[0]


def metrics(y, pr, grp):
    """grp: integer group ids = industry x as-of year, for the tercile spread."""
    ok = ~np.isnan(pr)
    y, pr, grp = y[ok], pr[ok], grp[ok]
    if len(y) < 5:
        return dict(rmse=np.nan, mae=np.nan, r2=np.nan, spearman=np.nan, spread=np.nan)
    e = y - pr
    sst = ((y - y.mean()) ** 2).sum()
    ry, rp = rankdata(y), rankdata(pr)
    sp = np.corrcoef(ry, rp)[0, 1] if rp.std() > 0 else np.nan
    num = den = 0.0
    for g in np.unique(grp):
        m = grp == g
        if m.sum() < 6:
            continue
        pct = (rankdata(pr[m]) - 1) / (m.sum() - 1)
        hi, lo = y[m][pct >= 2 / 3], y[m][pct <= 1 / 3]
        if len(hi) and len(lo):
            num += (hi.mean() - lo.mean()) * m.sum()
            den += m.sum()
    return dict(rmse=np.sqrt((e ** 2).mean()), mae=np.abs(e).mean(), r2=1 - (e ** 2).sum() / sst if sst > 0 else np.nan,
                spearman=sp, spread=num / den if den else np.nan)


def boot(y, pr, country, grp, rng):
    """Country-clustered bootstrap CI of every metric from fixed out-of-fold predictions."""
    ok = ~np.isnan(pr)
    cs = np.unique(country[ok])
    idx = {c: np.where((country == c) & ok)[0] for c in cs}
    draws = []
    for _ in range(B):
        sel = np.concatenate([idx[c] for c in rng.choice(cs, len(cs), replace=True)])
        draws.append(metrics(y[sel], pr[sel], grp[sel]))
    d = pd.DataFrame(draws)
    return d.quantile(0.025), d.quantile(0.975)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for spec in SPECS:
        p, trail = load(spec)
        grp = pd.factorize(p.industry + p.as_of_year.astype(str))[0]
        y, country = p.y.values, p.country.values
        for scheme in ("country", "region", "time"):
            preds = {}
            test_idx = []
            for tr, te in folds(p, scheme):
                if len(tr) < 20 or len(te) < 5:
                    continue
                test_idx.append(te)
                for k, v in predict(p.iloc[tr], p.iloc[te], trail).items():
                    preds.setdefault(k, np.full(len(p), np.nan))[te] = v
            if not test_idx:
                continue
            for k, pr in preds.items():
                m = ~np.isnan(pr)
                est = metrics(y, pr, grp)
                lo, hi = boot(y, pr, country, grp, rng)
                if k in CONSTANT:  # no rank information: report n/a, not the leave-fold-out-mean artifact
                    for kk in ("spearman", "spread"):
                        est[kk], lo[kk], hi[kk] = np.nan, np.nan, np.nan
                rows.append({"spec": spec, "cv": scheme, "model": k, "n_rows": int(m.sum()), "n_countries": len(set(country[m])),
                             **{kk: v for kk, v in est.items()}, **{f"{kk}_lo": lo[kk] for kk in est}, **{f"{kk}_hi": hi[kk] for kk in est}})
        print("done", spec, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(vp("reports/baselines.csv"), index=False)
    return df


if __name__ == "__main__":
    main()

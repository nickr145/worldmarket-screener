"""Freeze the shipped models: refit exactly as score_cells does, pickle them, verify they reproduce scored_cells.csv, write a manifest.
python -m screener.score.freeze -> models/shipped_v1_1/ (models/shipped_v1/ with WM_TAG=)"""
import hashlib
import json
import os
import shutil
import subprocess

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from screener.features.run_panel import MODELS, ROOT, vp
from screener.model.baselines import ALPHAS, SHARED, SPECIFIC, load
from screener.score.score_cells import LAMBDA, LEVEL, PRIMARY

OUT = MODELS


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab, _ = load(PRIMARY)
    fitted, meta = {}, {}
    for ind in sorted(lab.industry.unique()):
        t = lab[lab.industry == ind]
        cols = SPECIFIC[ind] + SHARED + ["trailing_growth"]
        gs = GridSearchCV(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge()), {"ridge__alpha": ALPHAS},
                          scoring="neg_mean_squared_error", cv=GroupKFold(n_splits=min(5, t.country.nunique())))
        gs.fit(t[cols], t.y, groups=t.country)
        fitted[ind] = gs.best_estimator_
        meta[ind] = {"features": cols, "alpha": float(gs.best_params_["ridge__alpha"]), "n_train": len(t), "industry_mean": float(t.y.mean())}
        joblib.dump(gs.best_estimator_, OUT / f"ridge_full_{ind}.joblib")
    # verify the frozen models reproduce the shipped growth point estimates
    sc = pd.read_csv(vp("reports/scored_cells.csv"))
    panel = pd.read_parquet(vp("data/panel/panel.parquet")).assign(log_gdp_pc=lambda d: np.log(d.gdp_pc_usd))
    err = []
    for r in sc[~sc.insufficient_data].itertuples():
        row = panel[(panel.country == r.country) & (panel.industry == r.industry) & (panel.as_of_year == r.scored_asof_year)]
        rf = fitted[r.industry].predict(row[meta[r.industry]["features"]])[0]
        im = meta[r.industry]["industry_mean"]
        err.append(abs(im + LAMBDA * (rf - im) - r.model_estimate))
    assert max(err) < 1e-9, f"frozen models do not reproduce scored_cells (max err {max(err)})"
    manifest = {
        "version": OUT.name.replace("shipped_", "shipped-").replace("_", "."),
        "frozen_at_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "final_commit": "the commit this release tag points to (a commit cannot record its own hash): git rev-parse <version>^{commit}",
        "growth": {"ranking_signal": "trailing real growth persistence", "point": f"industry_mean + {LAMBDA} * (ridge_full - industry_mean)",
                   "shrinkage_lambda": LAMBDA, "interval": f"{int(LEVEL*100)}% split-conformal, industry-specific, leave-country-out residuals",
                   "ridge": meta, "n_labelled_rows": len(lab), "labelled_as_of_years": sorted(map(int, lab.as_of_year.unique()))},
        "downside_deviation": {"ranking_signal": "trailing 12m downside deviation persistence", "point": "persistence", "interval_half_width": None},
        "max_drawdown": "ordering only, no estimate",
        "verdict_rule_v2": {"avoid": "downside-deviation signal in top third within industry (pct >= 2/3)",
                            "expand": "growth signal in top third and not avoid", "consider": "otherwise",
                            "forced_consider": ["Turkey (low confidence, not trained on)"], "thresholds_frozen": True},
        "input_hashes": {"panel": sha(vp("data/panel/panel.parquet")), "price_targets": sha(vp("data/panel/price_targets.parquet")),
                         "scored_cells": sha(vp("reports/scored_cells.csv")),
                         "currency_overrides": sha(os.environ.get("CCY_OVERRIDES", ROOT / "data/panel/currency_overrides.csv"))},
        "max_abs_repro_error_growth_point": float(max(err)),
    }
    sd = sc[~sc.insufficient_data]
    manifest["growth"]["conformal_half_width_by_industry"] = sd.groupby("industry").model_estimate_halfwidth.first().to_dict()
    manifest["downside_deviation"]["interval_half_width"] = float((sd.recent_downside_estimate_hi - sd.recent_downside_estimate).dropna().iloc[0]) if sd.recent_downside_estimate_hi.notna().any() else None
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1, default=str)
    for f in ("settings.yaml", "countries.yaml", "currencies.yaml", "macro.yaml"):
        shutil.copy(ROOT / "screener/config" / f, OUT / f)
    shutil.copytree(ROOT / "screener/config/industries", OUT / "industries", dirs_exist_ok=True)
    shutil.copy(vp("reports/scored_cells.csv"), OUT / "scored_cells_frozen.csv")
    print("frozen:", sorted(p.name for p in OUT.iterdir()), "| max repro error", manifest["max_abs_repro_error_growth_point"])


if __name__ == "__main__":
    main()

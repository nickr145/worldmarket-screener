"""Data and model helpers for the dashboard (no Streamlit imports, so they are unit-testable)."""
import json
from functools import lru_cache

import datetime as dt
import joblib
import numpy as np
import pandas as pd
import yaml

from screener.features.run_panel import MODELS, ROOT, vp
from screener.model.baselines import SPECIFIC

MODEL_VERSION = "shipped-v1.1"
DECISION_DATE = "30 Sep 2026"


def _skill():
    """Out-of-fold Spearman of the shipped ranking signals, read from the evaluation files (growth: primary spec; downside: confirmatory set)."""
    b, r = pd.read_csv(vp("reports/baselines.csv")), pd.read_csv(vp("reports/risk_targets.csv"))
    g = b[b.spec.str.startswith("PRIMARY") & (b.cv == "country") & (b.model == "trailing_persistence")].spearman.iloc[0]
    d = r[r.set.str.startswith("confirmatory") & (r.cv == "country") & (r.target == "dd_dev_12m") & (r.model == "trailing_persistence")].spearman.iloc[0]
    return f"{g:.2f}", f"{d:.2f}"


SKILL_GROWTH, SKILL_DOWNSIDE = _skill()

def _fmt(d: str) -> str:
    return dt.date.fromisoformat(d).strftime("%-d %b %Y")


_SNAP = json.load(open(vp("reports/data_snapshot.json")))
SNAPSHOT_DATE = _fmt(_SNAP["snapshot_date"])
LATEST_PRICE_DATE = _fmt(_SNAP["latest_price_date"])
PRETTY = {"account_ownership_pct": "Account ownership (% adults)", "credit_private_pct_gdp": "Private credit (% of GDP)", "bank_npl_pct": "Bank bad loans (%)",
          "mobile_subs_per_100": "Mobile subscriptions per 100", "wgi_reg_quality": "Regulatory quality index", "internet_users_pct": "Internet users (%)",
          "secure_servers_per_million": "Secure servers per million", "ict_service_exports_pct": "ICT service exports (% of exports)", "rd_pct_gdp": "R&D (% of GDP)",
          "urban_pop_pct": "Urban population (%)", "household_consumption_pct_gdp": "Household consumption (% of GDP)", "lpi_overall": "Logistics performance index",
          "health_exp_pct_gdp": "Health spending (% of GDP)", "oop_share_pct": "Out-of-pocket share of health spending (%)", "pop_65plus_pct": "Population aged 65+ (%)",
          "physicians_per_1000": "Physicians per 1,000", "life_expectancy": "Life expectancy", "log_gdp_pc": "Log GDP per capita", "firms_per_gdp": "Listed firms per $bn GDP",
          "trailing_growth": "Recent growth of listed peers"}
VERDICT_COLORS = {"expand": "#1b7f5c", "consider": "#e8c547", "avoid": "#c0392b", "insufficient_data": "#9aa0a6", "not_rated": "#b8c4d6"}


@lru_cache
def manifest() -> dict:
    return json.load(open(MODELS / "manifest.json"))


@lru_cache
def countries() -> dict:
    return yaml.safe_load(open(ROOT / "screener/config/countries.yaml"))["countries"]


@lru_cache
def panel() -> pd.DataFrame:
    return pd.read_parquet(vp("data/panel/panel.parquet")).assign(log_gdp_pc=lambda d: np.log(d.gdp_pc_usd))


def contributions(country: str, industry: str, year: int) -> dict:
    """Exact linear contributions of the ridge (with trailing growth) to its pre-shrinkage prediction:
    intercept + sum(coef_j * standardised x_j) = ridge prediction. Descriptive, not causal."""
    pipe = joblib.load(MODELS / f"ridge_full_{industry}.joblib")
    cols = manifest()["growth"]["ridge"][industry]["features"]
    p = panel()
    row = p[(p.country == country) & (p.industry == industry) & (p.as_of_year == year)]
    X = row[cols]
    imp, sc, rg = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
    z = sc.transform(imp.transform(X))[0]
    contrib = z * rg.coef_
    pred = float(rg.intercept_ + contrib.sum())
    assert abs(pred - pipe.predict(X)[0]) < 1e-9
    im = manifest()["growth"]["ridge"][industry]["industry_mean"]
    df = pd.DataFrame({"feature": cols, "label": [PRETTY.get(c, c) for c in cols], "value": X.iloc[0].values, "contribution": contrib})
    return {"table": df, "intercept": float(rg.intercept_), "ridge_prediction": pred, "industry_mean": im, "shrunk_estimate": im + 0.5 * (pred - im)}


def macro_age(country: str, industry: str, year: int) -> tuple[float, str]:
    """Oldest macro input (years between the as-of year and the observation) among the ridge's macro features."""
    row = panel()[(panel().country == country) & (panel().industry == industry) & (panel().as_of_year == year)].iloc[0]
    feats = [c for c in SPECIFIC[industry] + ["gdp_pc_usd"] if f"age_{c}" in row.index and pd.notna(row.get(f"age_{c}"))]
    if not feats:
        return float("nan"), ""
    worst = max(feats, key=lambda c: row[f"age_{c}"])
    return float(row[f"age_{worst}"]), PRETTY.get(worst, worst)


@lru_cache
def _firm_list() -> pd.DataFrame:
    return pd.read_csv(vp("data/panel/comparable_firms.csv"))


def comparable_firms(country: str, industry: str, year: int) -> pd.DataFrame:
    """Names and tickers of the listed firms in a cell (committed file, no financials)."""
    f = _firm_list()
    f = f[(f.country == country) & (f.industry == industry) & (f.as_of_year == year)]
    return pd.DataFrame({"Firm": f.name.values, "Ticker": f.ticker.values})


def insufficient_reason(r) -> str:
    n = r.get("n_members")
    return (f"only {int(n)} listed firms with usable financials (at least 5 needed)" if pd.notna(n) else "too few listed firms with usable financials (at least 5 needed)")


def client():
    """Anthropic client for the parser and narrator (patched in tests)."""
    from screener.parse.parser import get_client
    return get_client()


def md_section(path, heading_prefix: str) -> str:
    """Text of the markdown section whose '## ' heading starts with heading_prefix (until the next '## ')."""
    lines, out, on = open(path).read().splitlines(), [], False
    for ln in lines:
        if ln.startswith("## "):
            if on:
                break
            on = ln[3:].startswith(heading_prefix)
        if on:
            out.append(ln)
    return "\n".join(out)

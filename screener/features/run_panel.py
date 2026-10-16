"""Load caches -> firm-year table -> panel (data/panel/panel.parquet) + reports. Usage: python -m screener.features.run_panel"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from screener.features.build_panel import build_panel

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "screener" / "config"
TAG = os.environ.get("WM_TAG", "_v1_1")  # suffix on every generated file name; WM_TAG= (empty) rebuilds the v1 names
MODELS = ROOT / "models" / ("shipped_v1_1" if TAG else "shipped_v1")


def vp(rel):
    """ROOT / rel with TAG inserted before the extension (generated files only)."""
    p = ROOT / rel
    return p.with_name(p.stem + TAG + p.suffix)


def load_cfg():
    st = yaml.safe_load(open(CFG / "settings.yaml"))
    return st, yaml.safe_load(open(CFG / "countries.yaml"))["countries"], yaml.safe_load(open(CFG / "currencies.yaml"))


def load_macro():
    m = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "data/raw/macro/*.parquet")))])
    return m


def by_ccy(macro, name_primary, ccys, key, fallback=None):
    """{(ccy, year): value} for a macro series via currencies.yaml[ccy][key]; optional fallback (name) when missing."""
    out = {}
    for ccy, m in ccys.items():
        for nm in [name_primary] + ([fallback] if fallback else []):
            d = macro[(macro.name == nm) & (macro.iso3 == m[key])]
            for y, v in zip(d.year, d.value):
                out.setdefault((ccy, y), v)
    return out


def prepare(st, countries, ccys, macro, variant=""):
    fin = pd.concat([x for x in (pd.read_parquet(f) for f in glob.glob(str(ROOT / "data/raw/financials/*.parquet"))) if len(x)])
    from screener.ingest.financials import universe  # restrict to the CURRENT fetch universe (no stale/biotech cache files)
    keep = universe("_biotech" if variant == "biotech" else "")[0]
    if variant == "no_assumed":
        keep = keep[~keep.domicile_assumed]
    fin = fin[fin.symbol.isin(keep.symbol)]
    ov = pd.read_csv(os.environ.get("CCY_OVERRIDES", ROOT / "data/panel/currency_overrides.csv"))  # reviewed per-firm currency corrections
    m = dict(zip(ov.symbol, ov.corrected_ccy))
    fin["currency"] = [m.get(s_, c_) for s_, c_ in zip(fin.symbol, fin.currency)]
    fin = fin.sort_values("fiscal_year_end").drop_duplicates(["symbol", "fiscal_year"], keep="last")
    fx = by_ccy(macro, "fx_lcu_per_usd", ccys, "fx")
    fx.update({("USD", y): 1.0 for y in range(2000, 2031)})
    infl = by_ccy(macro, "inflation_pct", ccys, "cpi", fallback="imf_inflation_pct")
    fin["fx"] = [fx.get((c, y), np.nan) for c, y in zip(fin.currency, fin.fiscal_year)]
    fin["rev_usd"] = fin.revenue / fin.fx
    fin["infl"] = [infl.get((c, y), np.nan) / 100 for c, y in zip(fin.currency, fin.fiscal_year)]
    fin["deflate"] = [not countries[c].get("hyperinflation_accounting", False) for c in fin.country]
    EURO = {"DE", "FR", "ES", "IT", "NL"}  # EUR is the local currency here
    fin["non_local_ccy"] = [not ((cur == "EUR" and c in EURO) or ccys.get(cur, {}).get("fx") == countries[c]["iso3"]) for c, cur in zip(fin.country, fin.currency)]
    return fin, fx


def prepare_cells(st, countries, fx, variant=""):
    fu = pd.read_csv(ROOT / f"reports/firm_universe{'_biotech' if variant == 'biotech' else ''}.csv")
    fu = fu[fu.domestic]
    if variant == "no_assumed":
        fu = fu[~fu.domicile_assumed]
    fu["mcap_usd"] = [m / fx.get((c, 2025), np.nan) for m, c in zip(fu.mcap, fu.ccy)]
    cc = fu.groupby(["country", "industry"]).agg(n_listed=("symbol", "size"), mcap_usd=("mcap_usd", "sum")).reset_index()
    cc = cc.merge(cc.groupby("country").n_listed.sum().rename("country_listed").reset_index())
    cc = cc[(cc.n_listed >= st["min_cell_firms"]) & ~cc.country.isin(st["drop_from_panel"])]
    cc["iso3"] = cc.country.map(lambda c: countries[c]["iso3"])
    cc["region"] = cc.country.map(lambda c: countries[c]["region"])
    cc["low_confidence"] = cc.country.isin(st["low_confidence_countries"])
    cc["train_eligible"] = ~cc.country.isin(st["exclude_from_training"])
    return cc[["country", "industry", "iso3", "region", "n_listed", "mcap_usd", "country_listed", "low_confidence", "train_eligible"]]


def main(variant=""):
    st, countries, ccys = load_cfg()
    macro = load_macro()
    F, fx = prepare(st, countries, ccys, macro, variant)
    cells = prepare_cells(st, countries, fx, variant)
    if not variant:
        from screener.clean.currency_check import report
        report(F, fx, countries, ccys, st)
    macro = macro[macro.iso3.isin([m["iso3"] for m in countries.values()])]
    panel, obs = build_panel(F, macro, cells, st, st["as_of_years"])
    out = ROOT / "data/panel"
    out.mkdir(parents=True, exist_ok=True)
    suf = f"_{variant}" if variant else ""
    panel.to_parquet(vp(f"data/panel/panel{suf}.parquet"))
    obs.to_parquet(vp(f"data/panel/firm_obs{suf}.parquet"))
    F.to_parquet(vp("data/interim/firm_years.parquet")) if (ROOT / "data/interim").exists() else None
    return panel, obs, F


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "")

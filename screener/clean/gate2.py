"""Gate 2: a cell is usable only if >=5 firms have >=4 consecutive fiscal years of revenue > 0 AND CPI exists for the
country over those years. Writes reports/gate2_cells.csv (per-cell firm counts + as-of years supported) and prints the report.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "screener" / "config"


def load():
    from screener.features.run_panel import load_cfg, load_macro, prepare
    st, countries, ccys = load_cfg()
    macro = load_macro()
    F, fx = prepare(st, countries, ccys, macro)
    return st, countries, ccys, F


def runs_ok(years, k):
    best = cur = 0
    for y in range(min(years), max(years) + 1) if years else []:
        cur = cur + 1 if y in years else 0
        best = max(best, cur)
    return best >= k


def main(no_biotech=False, out="gate2_cells.csv"):
    st, countries, ccys, F = load()
    fu = pd.read_csv(ROOT / "reports" / "firm_universe.csv")
    fu = fu[fu.domestic & ~fu.country.isin(st["drop_from_panel"])]
    if no_biotech:
        fu = fu[fu.yf_industry != "Biotechnology"]
    n_min, k = st["min_cell_firms"], st["min_fy_consecutive"]
    # cell membership: domestic firms with >= n_min per cell, then top fetch_top_n by current cap (what was fetched)
    sizes = fu.groupby(["country", "industry"]).size()
    fu = fu[fu.set_index(["country", "industry"]).index.map(sizes) >= n_min]
    fu = fu.sort_values("mcap", ascending=False).groupby(["country", "industry"]).head(st["fetch_top_n"])
    F = F[F.symbol.isin(fu.symbol)]
    # a fiscal year counts for a firm only if revenue > 0 AND the inflation rate of its REPORTING currency exists for that year
    # (TR/AR are not deflated, so they need no CPI)
    good = F[(F.revenue > 0) & (F.infl.notna() | ~F.deflate)]
    fy_ok = good.groupby("symbol").fiscal_year.agg(set).to_dict()
    rows = []
    for (c, ind), g in fu.groupby(["country", "industry"]):
        ys = [fy_ok.get(s, set()) for s in g.symbol]
        n4 = sum(runs_ok(y, k) for y in ys)
        nl = F[F.symbol.isin(g.symbol)].drop_duplicates("symbol").non_local_ccy
        rows.append({"country": c, "industry": ind, "region": countries[c]["region"], "firms_fetched": len(g),
                     "firms_with_financials": sum(bool(y) for y in ys), "firms_4fy_revenue_and_cpi": n4,
                     "share_non_local_ccy": round(nl.mean(), 2) if len(nl) else np.nan,
                     "low_confidence": c in st["low_confidence_countries"] or countries[c].get("hyperinflation_accounting", False),
                     "van_firms": int((g.exchange == "VAN").sum()) if c == "CA" else 0, "usable": n4 >= n_min})
    t = pd.DataFrame(rows)
    from screener.features.run_panel import vp
    t.to_csv(vp(f"reports/{out}"), index=False)
    return t


def report(t, st, label):
    u = t[t.usable]
    per_c = u.groupby("country").size()
    print(f"\n=== {label} ===")
    print(f"(a) usable cells: {len(u)} of {len(t)} gate-1 cells | (b) countries with >=2: {(per_c >= 2).sum()} of {t.country.nunique()} | (c) by industry {u.groupby('industry').size().to_dict()}")
    print("   by region:", u.groupby("region").size().to_dict())
    return len(u), int((per_c >= 2).sum())


if __name__ == "__main__":
    st = yaml.safe_load(open(CFG / "settings.yaml"))
    full = main(False)
    a = report(full, st, "with Biotechnology")
    nb = main(True, "gate2_cells_no_biotech.csv")
    b = report(nb, st, "without Biotechnology")
    ch = full.merge(nb, on=["country", "industry"], how="outer", suffixes=("_bio", "_nobio"))
    print("cells whose usable flag changes:", ch[ch.usable_bio.fillna(False) != ch.usable_nobio.fillna(False)][["country", "industry"]].values.tolist())
    print(f"cost of dropping biotech: {a[0] - b[0]} cells; PASS" if a[0] >= st["gate2_min_cells"] else "STOP")

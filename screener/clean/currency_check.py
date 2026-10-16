"""Report-only anomaly check on firms' financial-currency tags (never changes data). Run on every panel build.

For each firm's latest known fiscal year it compares revenue, total assets, equity and net income per USD of market cap with the cell median
under the stored tag and under three other candidates (listing, home-country, USD). A tag is a CANDIDATE if tag != listing currency, the composite
(median log10 ratio) is >= 1.0 off under the tag, some other currency brings it within +/-0.3, the FX gap is >= 10x and >= 3 of 4 metrics are
within 0.5 log under that currency. REVIEW (looser, includes tag == listing) is for a human; corrections go in data/panel/currency_overrides.csv."""
import numpy as np
import pandas as pd

from screener.features.build_panel import decision_date
from screener.features.run_panel import ROOT

METRICS = ["revenue", "total_assets", "equity", "net_income"]
EURO = {"DE", "FR", "ES", "IT", "NL"}


def check(F, fx, countries, ccys, st, asof=2025):
    """F, fx as returned by run_panel.prepare. Returns one row per flagged firm."""
    fu = pd.read_csv(ROOT / "reports/firm_universe.csv")[["symbol", "ccy", "mcap"]].rename(columns={"ccy": "listing"})
    home = {}
    for c, m in ccys.items():
        home.setdefault(m["fx"], c)
    D = decision_date(asof, st)
    G = F[(F.revenue > 0) & (F.fiscal_year <= asof) & (F.as_of_date <= D)].sort_values("fiscal_year").groupby("symbol").tail(1).merge(fu, on="symbol")
    g = lambda c, y: 1.0 if c == "USD" else fx.get((c, y), np.nan)
    G["home"] = ["EUR" if c in EURO else home.get(countries[c]["iso3"]) for c in G.country]
    G["mcap_usd"] = G.mcap / [g(c, 2025) for c in G.listing]
    for m in METRICS:
        G[m] = G[m].where(G[m] > 0)
        G["r_" + m] = G[m] / [g(c, y) for c, y in zip(G.currency, G.fiscal_year)] / G.mcap_usd
        G["med_" + m] = G.groupby(["country", "industry"])["r_" + m].transform("median")
    rows = []
    for r in G.itertuples():
        def L(c, m):
            f, v, md = g(c, r.fiscal_year), getattr(r, m), getattr(r, "med_" + m)
            return np.log10(v / f / r.mcap_usd / md) if f == f and v == v and md == md and md > 0 and r.mcap_usd == r.mcap_usd else np.nan
        per = {c: {m: L(c, m) for m in METRICS} for c in dict.fromkeys([r.currency, r.listing, r.home, "USD"]) if isinstance(c, str) and g(c, r.fiscal_year) == g(c, r.fiscal_year)}
        comp = {c: np.nanmedian(list(v.values())) for c, v in per.items() if np.isfinite(list(v.values())).sum() >= 2}
        if r.currency not in comp or abs(comp[r.currency]) < 1.0:
            continue
        best = min(comp, key=lambda c: abs(comp[c]))
        if best == r.currency:
            continue
        gap = max(g(best, r.fiscal_year), g(r.currency, r.fiscal_year)) / min(g(best, r.fiscal_year), g(r.currency, r.fiscal_year))
        n_ok = sum(abs(per[best][m]) <= 0.5 for m in METRICS if np.isfinite(per[best][m]))
        n_better = sum(abs(per[best][m]) < abs(per[r.currency][m]) - 0.5 for m in METRICS if np.isfinite(per[best][m]) and np.isfinite(per[r.currency][m]))
        strict = r.currency != r.listing and abs(comp[best]) <= 0.3 and gap >= 10 and n_ok >= 3
        if strict or (abs(comp[best]) <= 0.5 and n_better >= 2 and gap >= 10):
            rows.append(dict(symbol=r.symbol, country=r.country, industry=r.industry, tag=r.currency, listing=r.listing, closest=best,
                             composite_tag=round(comp[r.currency], 2), composite_closest=round(comp[best], 2), fx_gap=round(gap, 1), metrics_ok=n_ok,
                             level="CANDIDATE" if strict else "REVIEW"))
    return pd.DataFrame(rows)


def report(F, fx, countries, ccys, st):
    out = check(F, fx, countries, ccys, st)
    print(f"currency-tag check (report only, nothing applied): {len(out)} flagged")
    if len(out):
        print(out.to_string(index=False))
    return out

"""Selection-bias sensitivity: cell median real growth (h=2) under (i) all fetched firms, (ii) top-25 by CURRENT market cap
(old rule), (iii) top-25 by FY t revenue (panel rule). Five big countries x two industries, as-of years 2022 and 2023."""
import pandas as pd

from screener.features.build_panel import decision_date, real_growth
from screener.features.run_panel import ROOT, load_cfg, load_macro, prepare, vp

st, countries, ccys = load_cfg()
F, fx = prepare(st, countries, ccys, load_macro())
fu = pd.read_csv(ROOT / "reports/firm_universe.csv").set_index("symbol")
fl = st["firm_growth_floor"]["min_prior_revenue_usd"]
fy_by = {s: {r["fiscal_year"]: r for r in g.to_dict("records")} for s, g in F.groupby("symbol")}
out = []
for c in ["US", "JP", "IN", "KR", "HK"]:
    for ind in ["healthcare", "software"]:
        g = F[(F.country == c) & (F.industry == ind)]
        for t in (2022, 2023):
            D = decision_date(t, st)
            known = g[(g.fiscal_year == t) & (g.as_of_date <= D) & (g.revenue > 0)].dropna(subset=["rev_usd"])
            gr = {s: real_growth(fy_by[s], t, 2, fl, bool(fy_by[s][t]["deflate"])) for s in known.symbol}
            gr = pd.Series(gr).dropna()
            cap = fu.loc[known.symbol, "mcap"]
            top_cap = cap.sort_values(ascending=False).head(25).index
            top_rev = known.sort_values("rev_usd", ascending=False).head(25).symbol
            out.append({"country": c, "industry": ind, "as_of": t, "n_fetched": len(known), "n_valid": len(gr),
                        "all_fetched": gr.median(), "top25_current_cap": gr.reindex(top_cap).median(), "top25_fy_t_revenue": gr.reindex(top_rev).median()})
r = pd.DataFrame(out)
r["bias_capsel_minus_revsel_pp"] = (r.top25_current_cap - r.top25_fy_t_revenue) * 100
r.to_csv(vp("reports/selection_sensitivity.csv"), index=False)
pd.set_option("display.width", 250)
print(r.round(3).to_string(index=False))

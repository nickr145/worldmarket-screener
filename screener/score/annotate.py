"""Display-only annotations added to reports/scored_cells.csv after the freeze (no model output or verdict changes):
freshness-bias check, lower_reliability flag, country names. python -m screener.score.annotate -> reports/freshness_bias.csv"""
import numpy as np
import pandas as pd

from screener.features.build_panel import decision_date
from screener.features.run_panel import ROOT, vp, load_cfg, load_macro, prepare

NEW = ["country_name", "n_members_2024", "n_dropped_missing_fy2025", "fy2025_drop_share", "drop_flag", "median_growth_dropped", "median_growth_retained",
       "lower_reliability", "lower_reliability_reason"]


def main():
    st, countries, ccys = load_cfg()
    sc = pd.read_csv(vp("reports/scored_cells.csv")).drop(columns=NEW, errors="ignore")
    F, _ = prepare(st, countries, ccys, load_macro())
    D = decision_date(2025, st)
    has25 = set(F[(F.fiscal_year == 2025) & (F.revenue > 0) & (F.as_of_date <= D)].symbol)
    obs = pd.read_parquet(vp("data/panel/firm_obs.parquet"))
    m24 = obs[obs.t == 2024].copy()
    m24["dropped"] = ~m24.symbol.isin(has25)
    rows = []
    for (c, i), g in m24.groupby(["country", "industry"]):
        d, r = g[g.dropped], g[~g.dropped]
        rows.append({"country": c, "industry": i, "n_members_2024": len(g), "n_dropped_missing_fy2025": len(d), "fy2025_drop_share": len(d) / len(g),
                     "median_growth_dropped": d.trail.median() if len(d) else np.nan, "median_growth_retained": r.trail.median() if len(r) else np.nan,
                     "n_retained": len(r)})
    fb = pd.DataFrame(rows)
    fb["drop_flag"] = fb.fy2025_drop_share >= 0.20
    n25 = pd.read_parquet(vp("data/panel/panel.parquet")).query("as_of_year == 2025").set_index(["country", "industry"]).n_members
    fb["n_members_2025"] = [n25.get((c, i), np.nan) for c, i in zip(fb.country, fb.industry)]
    fb["fell_below_5_in_2025"] = fb.n_members_2025 < st["min_cell_firms"]
    fb.to_csv(vp("reports/freshness_bias.csv"), index=False)
    out = sc.merge(fb[["country", "industry", "n_members_2024", "n_dropped_missing_fy2025", "fy2025_drop_share", "drop_flag", "median_growth_dropped",
                       "median_growth_retained"]], on=["country", "industry"], how="left")
    out["country_name"] = out.country.map(lambda c: countries[c]["name"])
    # lower_reliability: the industry with the widest growth interval AND the widest coverage CI AND the fewest scored cells
    sd = out[~out.insufficient_data]
    ind = pd.DataFrame({"half_width": sd.groupby("industry").model_estimate_halfwidth.first(), "n_cells": sd.groupby("industry").size()})
    weakest = ind.half_width.idxmax() if ind.half_width.idxmax() == ind.n_cells.idxmin() else None
    out["lower_reliability"] = out.industry == weakest
    out["lower_reliability_reason"] = np.where(out.lower_reliability, "widest growth interval of the four industries, widest coverage CI, fewest scored cells", "")
    out.to_csv(vp("reports/scored_cells.csv"), index=False)
    return fb, out, ind, weakest


if __name__ == "__main__":
    fb, out, ind, weakest = main()
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print("firms dropped from 2025 membership for missing FY2025 (of the as-of-2024 members):")
    print(f"  total dropped {int(fb.n_dropped_missing_fy2025.sum())} of {int(fb.n_members_2024.sum())} ({fb.n_dropped_missing_fy2025.sum() / fb.n_members_2024.sum():.1%}); cells with any drop {(fb.n_dropped_missing_fy2025 > 0).sum()} of {len(fb)}")
    print("  cells that fell below 5 firms in 2025:", fb[fb.fell_below_5_in_2025][["country", "industry", "n_members_2024", "n_dropped_missing_fy2025", "n_members_2025"]].to_dict("records"))
    print("  cells with drops >= 20% of FY2024 membership:", fb[fb.drop_flag][["country", "industry", "n_members_2024", "n_dropped_missing_fy2025"]].to_dict("records"))
    # firm-level pooled and by industry
    obs = pd.read_parquet(vp("data/panel/firm_obs.parquet"))
    st, countries, ccys = load_cfg()
    F, _ = prepare(st, countries, ccys, load_macro())
    has25 = set(F[(F.fiscal_year == 2025) & (F.revenue > 0) & (F.as_of_date <= decision_date(2025, st))].symbol)
    m = obs[obs.t == 2024].assign(dropped=lambda d: ~d.symbol.isin(has25))
    t = m.groupby("dropped").trail.agg(["count", "median"]).rename(index={False: "retained", True: "dropped"})
    print("\nmedian FY2024 real growth (firm level, winsorised): pooled"); print(t.round(4).to_string())
    byi = m.groupby(["industry", "dropped"]).trail.agg(["count", "median"]).unstack().round(4)
    byi.columns = [f"{a}_{'dropped' if b else 'retained'}" for a, b in byi.columns]
    print("\nby industry:"); print(byi.to_string())
    print("\nlower_reliability industry:", weakest, "\n", ind.round(3).to_string())

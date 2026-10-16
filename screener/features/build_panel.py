"""Panel builder: one row per (country, industry, as_of_year t).

Decision date D_t = Sep 30 of year t+1 (settings.asof_decision). It is the earliest date by which every firm's FY t
(fiscal_year rule in clean/fiscal.py, FYE up to May t+1, +lag) is public. A figure is *known* at D_t only if its
as_of_date (fiscal_year_end + filing lag; macro: year-end + release lag) <= D_t. The builder never looks at
fiscal_year_end for that decision. Targets use FY > t only. Snapshot columns (SNAPSHOT_COLS) are current counts / market caps
and are NOT as-of dated; they are exempt from the audit and documented as such.
"""
import numpy as np
import pandas as pd

SNAPSHOT_COLS = ["n_listed_cell", "firms_per_gdp", "firms_per_capita", "cell_mktcap_to_gdp", "country_listed_4ind_per_gdp"]
TARGETS = ["target_h1", "target_h2", "target_h1_usd", "target_h2_usd"]


def decision_date(t, cfg):
    d = cfg["asof_decision"]
    return pd.Timestamp(year=t + d["year_offset"], month=d["month"], day=d["day"])


def cagr(a, b, h):
    return (b / a) ** (1 / h) - 1 if a and b and a > 0 and b > 0 else np.nan


def real_growth(fy, t0, h, floor, deflate, usd=False):
    """CAGR of revenue from FY t0 to t0+h. Real = deflated by the reporting currency's inflation; usd = in USD."""
    a, b = fy.get(t0), fy.get(t0 + h)
    if a is None or b is None or not (a["rev_usd"] >= floor):  # base-year revenue floor (USD-equivalent)
        return np.nan
    if usd:
        return cagr(a["rev_usd"], b["rev_usd"], h)
    f = 1.0
    if deflate:
        for k in range(1, h + 1):
            p = fy.get(t0 + k, {}).get("infl", np.nan)
            f *= 1 + p
    if np.isnan(f):
        return np.nan
    return cagr(a["revenue"], b["revenue"] / f, h)


def winsor(s, lo, hi):
    s = s.dropna()
    return s.clip(s.quantile(lo), s.quantile(hi)) if len(s) else s


def macro_asof(macro, iso3, D):
    """Latest macro value per indicator with as_of_date <= D, plus age (years between D's data year and obs year)."""
    m = macro[(macro.iso3 == iso3) & (macro.as_of_date <= D)]
    last = m.sort_values("year").groupby("name").tail(1)
    out = {r["name"]: r["value"] for _, r in last.iterrows()}
    out.update({f"age_{r['name']}": D.year - 1 - r["year"] for _, r in last.iterrows()})
    return out, (last.as_of_date.max() if len(last) else pd.NaT)


def build_panel(F, macro, cells, cfg, years):
    """F: firm-year table (see module doc). cells: one row per gate-1 cell. Returns (panel, firm_obs)."""
    fy_by = {s: {r["fiscal_year"]: r for r in g.to_dict("records")} for s, g in F.groupby("symbol")}
    rows, obs = [], []
    for t in years:
        D = decision_date(t, cfg)
        # firm-level growth for every firm known at D, winsorised per as-of year across cells
        per_cell = {}
        for c in cells.itertuples():
            g = F[(F.country == c.country) & (F.industry == c.industry)]
            known = g[(g.fiscal_year == t) & (g.as_of_date <= D) & (g.revenue > 0)].dropna(subset=["rev_usd"])
            known = known[known.rev_usd >= cfg.get("min_asof_revenue_usd", 0)]  # as-of FY t revenue floor for membership
            top = known.sort_values("rev_usd", ascending=False).head(cfg["top_n_per_cell"])
            per_cell[(c.country, c.industry)] = top
        recs = []
        for (cn, ind), top in per_cell.items():
            for r in top.itertuples():
                fy = {k: v for k, v in fy_by[r.symbol].items()}
                fl = cfg["firm_growth_floor"]["min_prior_revenue_usd"]
                dfl = bool(r.deflate)
                known_hist = {k: v for k, v in fy.items() if k <= t and v["as_of_date"] <= D}  # features: known at D only
                recs.append({"symbol": r.symbol, "country": cn, "industry": ind, "t": t,
                             "trail": real_growth({**known_hist}, t - 1, 1, fl, dfl),
                             **{f"h{h}": real_growth(fy, t, h, fl, dfl) for h in (1, 2)},
                             **{f"h{h}_usd": real_growth(fy, t, h, fl, dfl, usd=True) for h in (1, 2)},
                             "rev_usd": r.rev_usd,
                             "op_margin": r.operating_income / r.revenue, "net_margin": r.net_income / r.revenue,
                             "leverage": r.total_debt / r.total_assets if r.total_assets and r.total_assets > 0 else np.nan,
                             "max_asof": max([r.as_of_date] + ([fy[t - 1]["as_of_date"]] if t - 1 in known_hist else []))})
        if not recs:
            continue
        R = pd.DataFrame(recs)
        lo, hi = cfg["firm_growth_floor"]["winsorize"]
        for col in ["trail", "h1", "h2", "h1_usd", "h2_usd"]:
            R[col] = winsor(R[col], lo, hi).reindex(R.index)
        obs.append(R)
        for c in cells.itertuples():
            g = R[(R.country == c.country) & (R.industry == c.industry)]
            m, m_asof = macro_asof(macro, c.iso3, D)
            row = {"country": c.country, "industry": c.industry, "region": c.region, "as_of_year": t, "decision_date": D,
                   "n_members": len(g), "low_confidence": bool(c.low_confidence), "train_eligible": bool(c.train_eligible)}
            nmin = cfg["min_cell_firms"]
            for col, name in [("trail", "trailing_growth"), ("h1", "target_h1"), ("h2", "target_h2"),
                              ("h1_usd", "target_h1_usd"), ("h2_usd", "target_h2_usd")]:
                v = g[col].dropna()
                row[name] = v.median() if len(v) >= nmin else np.nan
                row[f"n_{name}"] = len(v)
            for col in ["op_margin", "net_margin", "leverage"]:
                row[f"{col}_med"] = g[col].median() if len(g) >= nmin else np.nan
            sh = g.rev_usd / g.rev_usd.sum() if len(g) else pd.Series(dtype=float)
            row["hhi"] = float((sh ** 2).sum()) if len(g) else np.nan
            row["top1_share"] = float(sh.max()) if len(g) else np.nan
            row["concentrated"] = bool(row["top1_share"] > cfg["concentration_flag"]) if len(g) else False
            row["status"] = "ok" if len(g) >= nmin else "insufficient_data"
            row["features_max_asof_date"] = max([x for x in [g.max_asof.max() if len(g) else pd.NaT, m_asof] if pd.notna(x)], default=pd.NaT)
            gdp = m.get("gdp_usd", np.nan) / 1e9
            pop = m.get("population", np.nan) / 1e6
            row.update({"n_listed_cell": c.n_listed, "firms_per_gdp": c.n_listed / gdp, "firms_per_capita": c.n_listed / pop,
                        "cell_mktcap_to_gdp": c.mcap_usd / m.get("gdp_usd", np.nan),
                        "country_listed_4ind_per_gdp": c.country_listed / gdp})
            row.update(m)
            rows.append(row)
    return pd.DataFrame(rows), (pd.concat(obs) if obs else pd.DataFrame())

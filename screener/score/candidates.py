"""Candidate table for a confirmed profile: scored cells of the profile's industry, minus home_country and existing_markets."""
import pandas as pd

from screener.score.labels import display_key, flag_icons, flag_keys

from screener.features.run_panel import ROOT, vp

ORDER_NOTE = ("Display order only (not a model): label (Strong momentum, volatility not elevated; Neutral; Elevated volatility; Not rated last), then unflagged before flagged "
              "(the thin-cell flag does not count), then rank (first label: recent growth rank; middle label: recent growth percentile minus downside percentile; "
              "last label: highest downside first).")


def candidate_table(industry: str, exclude: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (scored candidates in display order, cells that cannot be scored)."""
    sc = pd.read_csv(vp("reports/scored_cells.csv"))
    sc = sc[(sc.industry == industry) & ~sc.country.isin(exclude)].copy()
    unscored = sc[sc.insufficient_data | sc.verdict.isin(["insufficient_data"])]
    ok = sc.drop(unscored.index).copy()
    ok["display_key"] = ok.apply(display_key, axis=1)
    ok["_tier"] = ok.display_key.map({"expand": 0, "consider": 1, "avoid": 2, "not_rated": 3}).fillna(1)
    ok["flag_icons"] = ok.apply(flag_icons, axis=1)
    ok["n_flags"] = ok.apply(lambda r: len([k for k in flag_keys(r) if k != "thin"]), axis=1)
    ok["_flagged"] = (ok.n_flags > 0).astype(int)
    ok["_key"] = ok.apply(lambda r: (-r.recent_growth_pct, r.recent_downside_pct if pd.notna(r.recent_downside_pct) else 0.5) if r.verdict == "expand"
                          else (-(r.recent_growth_pct - (r.recent_downside_pct if pd.notna(r.recent_downside_pct) else 0.5)),) if r.verdict == "consider"
                          else (-(r.recent_downside_pct if pd.notna(r.recent_downside_pct) else 0),), axis=1)
    ok = ok.sort_values(["_tier", "_flagged", "_key"]).drop(columns=["_tier", "_flagged", "_key"]).reset_index(drop=True)
    return ok, unscored

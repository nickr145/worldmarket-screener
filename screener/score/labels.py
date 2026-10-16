"""Shared display vocabulary. Two different things, never conflated:
  recent_growth_signal : trailing real growth of listed peers (persistence). Used for rank and the verdict.
  model_estimate       : ridge (with trailing growth) shrunk toward the industry mean, with an 80% conformal interval.
Downside volatility: recent_downside_signal (trailing downside deviation; persistence is also its estimate, with an interval)."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# UI display names. Internal tier names (expand/consider/avoid) stay frozen in CSVs, tests and the registered sections.
DISPLAY = {"expand": "Strong momentum, volatility not elevated", "consider": "Neutral", "avoid": "Elevated volatility"}
DISPLAY_INSUFFICIENT = "Insufficient data"
NOT_RATED = "Not rated (reference only)"  # shown instead of the label for low-confidence cells (Turkey); the internal tier stays 'consider'
INTERNAL = {**{v: k for k, v in DISPLAY.items()}, NOT_RATED: "not_rated"}
RULES = {"expand": "Top third of the industry on recent growth, and not in the top third on recent downside volatility",
         "consider": "Not in the top third for recent growth and not in the top third for downside volatility; this includes markets with weak recent growth",
         "avoid": "Top third of the industry on recent downside volatility"}
NOT_RATED_RULE = "Turkey is scored for reference only (excluded from the model's training data); its internal tier is 'consider' but it is shown as Not rated"
TIER_NOTE = ("Tiers are thirds of the within-industry ranking, so two markets on either side of a cutoff (for example downside rank 16 vs 17 of 25) "
             "are not meaningfully different.")
INTERVAL_NOTE = ("Conformal half-widths are constant within an industry, so the displayed intervals do not distinguish more uncertain cells from less uncertain ones.")
UNADJUSTED = {c for c, m in yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config/countries.yaml"))["countries"].items() if m.get("hyperinflation_accounting")}
UNADJUSTED_GROWTH = "unadjusted for inflation (not CPI-deflated)"
THIN_N = 10


def is_not_rated(r) -> bool:
    return bool(r.get("low_confidence")) if pd.notna(r.get("low_confidence")) else False


def display_key(r) -> str:
    """Key of the displayed category: the internal tier, or 'not_rated' for low-confidence cells (display only)."""
    return "not_rated" if is_not_rated(r) else r["verdict"]


def display_label(r) -> str:
    return NOT_RATED if is_not_rated(r) else DISPLAY[r["verdict"]]


def growth_basis(r) -> str:
    """How the growth signal is deflated: 'real' (CPI-deflated), or the unadjusted wording for hyperinflation-accounting countries."""
    return UNADJUSTED_GROWTH if r["country"] in UNADJUSTED else "real"


def label_table() -> pd.DataFrame:
    """Mapping shown on the label-explainer and model-card pages (and mirrored in docs/model_card.md)."""
    return pd.DataFrame({"Display label": [DISPLAY[k] for k in ("expand", "consider", "avoid")], "Internal tier name": ["expand", "consider", "avoid"],
                         "Rule (within each industry)": [RULES[k] for k in ("expand", "consider", "avoid")]}
                        ).pipe(lambda t: pd.concat([t, pd.DataFrame({"Display label": [NOT_RATED], "Internal tier name": ["consider"], "Rule (within each industry)": [NOT_RATED_RULE]})], ignore_index=True))


OUTSIDE_NOTE = "Recent growth is unusually {dir} versus the model estimate; the model expects partial reversion."
RENAME = {
    "growth_signal": "recent_growth_signal", "growth_rank": "recent_growth_rank", "pct_growth": "recent_growth_pct",
    "growth_point": "model_estimate", "growth_lo": "model_estimate_lo", "growth_hi": "model_estimate_hi", "growth_q": "model_estimate_halfwidth",
    "growth_thr": "v1_growth_threshold", "dd_thr": "v1_downside_threshold",
    "dd_signal": "recent_downside_signal", "dd_rank": "recent_downside_rank", "pct_dd": "recent_downside_pct",
    "dd_point": "recent_downside_estimate", "dd_lo": "recent_downside_estimate_lo", "dd_hi": "recent_downside_estimate_hi",
}
ICONS = {"low_confidence": ("⚠", "low-confidence country (not trained on)"), "concentrated": ("◼", "one firm is over 60% of the cell's revenue"),
         "assumed": ("◊", "some listed firms have an assumed home country"), "late": ("…", "20% or more of last year's firms had not reported the latest year"),
         "lower_reliability": ("◔", "lower-reliability industry (widest intervals, fewest cells)"), "outside": ("↕", "recent growth is outside the model's interval"),
         "unreliable": ("≈", "growth intervals unreliable for this industry"), "stale": ("◌", "built from older data than other cells")}


ICONS["thin"] = ("‡", f"thin cell: fewer than {THIN_N} firms (display only)")


def _t(x):
    return bool(x) if pd.notna(x) else False


def flag_keys(r) -> list[str]:
    k = []
    if pd.notna(r.get("n_members")) and r.get("n_members") < THIN_N: k.append("thin")
    if _t(r.get("low_confidence")): k.append("low_confidence")
    if _t(r.get("concentrated")): k.append("concentrated")
    if pd.notna(r.get("domicile_assumed_share")) and r.get("domicile_assumed_share") > 0: k.append("assumed")
    if _t(r.get("drop_flag")) or _t(r.get("stale_firms_flag")): k.append("late")
    if _t(r.get("lower_reliability")): k.append("lower_reliability")
    if _t(r.get("signal_outside_interval")): k.append("outside")
    if pd.notna(r.get("interval_reliable")) and not _t(r.get("interval_reliable")): k.append("unreliable")
    if _t(r.get("stale_financials")): k.append("stale")
    return k


def flag_icons(r) -> str:
    return " ".join(ICONS[k][0] for k in flag_keys(r))


def outside_note(r) -> str:
    return OUTSIDE_NOTE.format(dir=r["signal_outside_direction"]) if _t(r.get("signal_outside_interval")) else ""


def add_outside(df, signal="growth_signal", lo="growth_lo", hi="growth_hi"):
    df["signal_outside_interval"] = (df[signal] < df[lo]) | (df[signal] > df[hi])
    df["signal_outside_direction"] = np.where(df[signal] > df[hi], "high", np.where(df[signal] < df[lo], "low", ""))
    return df

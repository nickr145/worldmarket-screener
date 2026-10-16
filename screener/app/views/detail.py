"""Market detail page."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from screener.app import services as S
from screener.parse.narrate import LABEL_DEFINITIONS, narrate
from screener.score.labels import ICONS, INTERVAL_NOTE, display_key, display_label, flag_keys, growth_basis, outside_note

BADGE = ('<span style="display:inline-block;padding:2px 8px;margin:2px 4px 2px 0;border-radius:10px;background:{bg};color:{fg};font-size:0.85em">{t}</span>')
KEY_TEXT = {"low_confidence": "Low-confidence country (Turkey): never actionable", "concentrated": "Concentrated: one firm over 60% of revenue",
            "assumed": "Assumed home country for some firms", "late": "Late reporters: 20%+ of last year's firms missing the latest year",
            "lower_reliability": "Lower-reliability industry", "outside": "Recent growth outside the model's interval", "unreliable": "Growth intervals unreliable",
            "stale": "Older data than other cells", "thin": "Thin cell: fewer than 10 firms"}


def all_keys(r):
    ks = flag_keys(r)
    if pd.notna(r.get("data_freshness")) and r.data_freshness < 0.60:
        ks.append("late")
    return list(dict.fromkeys(ks))


def key_text(k, r):
    """Badge text. Numbers a narration may quote (assumed share, concentration share) are shown here so they are visible on the page."""
    if k == "assumed":
        n = int(r.n_members)
        return f"Assumed home country for {round(r.domicile_assumed_share * n)} of {n} firms ({r.domicile_assumed_share:.0%})"
    if k == "concentrated":
        return f"Concentrated: one firm is {r.top1_share:.0%} of the cell's revenue (flag above 60%)"
    return KEY_TEXT[k]


def badges(r):
    ks = all_keys(r)
    html = "".join(BADGE.format(bg="#fdecea" if k in ("low_confidence", "concentrated") else "#fff4d6", fg="#333", t=f"{ICONS[k][0]} {key_text(k, r)}") for k in ks)
    return html or BADGE.format(bg="#e6f4ea", fg="#333", t="No reliability flags")


def contributions_chart(c):
    df = c["table"].sort_values("contribution")
    fig = go.Figure(go.Bar(x=df.contribution * 100, y=df.label, orientation="h", marker_color=["#c0392b" if v < 0 else "#1b7f5c" for v in df.contribution],
                           hovertemplate="%{y}: %{x:.2f} percentage points<extra></extra>"))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Contribution to the ridge prediction (percentage points of growth)")
    return fig


def render():
    st.header("Market detail")
    res = st.session_state.get("result")
    if not res or not len(res["cand"]):
        st.info("Score a company on the Screen page first, then open a market.")
        return
    cand = res["cand"].set_index("country", drop=False)
    country = st.selectbox("Market", cand.country.tolist(), format_func=lambda c: cand.country_name[c], key="detail_country")
    r = cand.loc[country]
    st.subheader(f"{r.country_name} · {r.industry.replace('_', ' ')}")
    dk = display_key(r)
    color = S.VERDICT_COLORS[dk]
    st.markdown(f'<span style="background:{color};color:{"#fff" if dk in ("expand", "avoid") else "#000"};padding:3px 12px;border-radius:12px;font-weight:700">{display_label(r)}</span> '
                f"&nbsp; {LABEL_DEFINITIONS[dk]}", unsafe_allow_html=True)
    st.markdown(badges(r), unsafe_allow_html=True)
    if r.low_confidence:
        st.warning("Turkey is scored for reference only: it is excluded from the model's training data, never receives a label, and its growth is "
                   "unadjusted for inflation (not CPI-deflated).")
    if r.signal_outside_interval is True or r.signal_outside_interval == True:
        st.info("↕ " + outside_note(r))
    a, b, c = st.columns(3)
    a.metric(f"Recent growth signal (listed peers, {growth_basis(r)})", f"{r.recent_growth_signal:.1%}", f"rank {int(r.recent_growth_rank)} of {int(r.n_ranked_in_industry)}" if pd.notna(r.recent_growth_rank) else "not ranked", delta_color="off")
    b.metric("Model estimate of growth", f"{r.model_estimate:.1%}", f"80% interval {r.model_estimate_lo:.1%} to {r.model_estimate_hi:.1%}", delta_color="off")
    if pd.notna(r.recent_downside_signal):
        c.metric("Recent downside volatility", f"{r.recent_downside_signal:.1%}",
                 f"rank {int(r.recent_downside_rank)} of {int(r.n_ranked_in_industry)} (1 = calmest) · 80% interval {r.recent_downside_estimate_lo:.1%} to {r.recent_downside_estimate_hi:.1%}"
                 if pd.notna(r.recent_downside_rank) else "not ranked", delta_color="off")
    else:
        c.metric("Recent downside volatility", "n/a", "no usable price history", delta_color="off")
    st.caption("The signal is what happened recently and drives the rank and label; the model estimate is a separate, shrunk estimate with an interval. Max drawdown is an ordering only and has no estimate. " + INTERVAL_NOTE)

    st.markdown("#### Data age and freshness")
    age, which = S.macro_age(r.country, r.industry, int(r.scored_asof_year))
    f = r.data_freshness
    x1, x2, x3 = st.columns(3)
    x1.metric("As-of fiscal year", int(r.scored_asof_year), f"decision date {S.DECISION_DATE}", delta_color="off")
    late = "late" in all_keys(r)  # same rule as the badge: "ok" only when no late-reporter flag applies
    x2.metric("Firms with the latest year reported", "n/a" if pd.isna(f) else f"{f:.0%}", "late reporters" if late else "ok", delta_color="off")
    x3.metric("Oldest macro input used", "n/a" if pd.isna(age) else f"{age:.0f} yr", which, delta_color="off")

    st.markdown("#### What the ridge model contributes (descriptive, not causal)")
    con = S.contributions(r.country, r.industry, int(r.scored_asof_year))
    st.plotly_chart(contributions_chart(con), width="stretch", key="contrib_chart")
    st.caption(f"Exact linear contributions: ridge intercept {con['intercept']:.1%} + the bars = ridge prediction {con['ridge_prediction']:.1%}; the model estimate then shrinks halfway toward the "
               f"industry mean ({con['industry_mean']:.1%}) to {con['shrunk_estimate']:.1%}. These are descriptive associations in 2023-2024 data. In testing, macro features added no skill over the industry mean; recent growth is the part that carries signal.")

    st.markdown("#### Plain-language description")
    key = (r.country, r.industry)
    if st.button("Describe this market", key="narrate_btn"):
        try:
            st.session_state.setdefault("narr", {})[key] = narrate(r, S.client())
        except Exception as e:
            st.error(f"Could not generate the description: {e}")
    n = st.session_state.get("narr", {}).get(key)
    if n:
        st.write(n.text)
        if n.source == "template":
            st.caption("⚙ **Template narration**: the language model's text did not pass the number-and-wording check, so a fixed template was used.")
        else:
            st.caption("Written by a language model from the numbers on this page only; every number is filled in from its labelled field and checked.")

    st.markdown("#### Listed firms in this cell")
    st.dataframe(S.comparable_firms(r.country, r.industry, int(r.scored_asof_year)), hide_index=True, width="stretch", key="comparables")
    st.caption(f"{int(r.n_members)} firms. Names and tickers only: the firms whose figures make up this cell for the as-of year. No financials are shown.")

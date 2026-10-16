"""Screen page: description -> editable parsed profile -> Score markets -> map + table."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from screener.app import services as S
from screener.parse.parser import ProfileDraft, parse_prompt, unsupported_message
from screener.parse.schema import INDUSTRIES
from screener.score.candidates import ORDER_NOTE, candidate_table
from screener.score.labels import DISPLAY, DISPLAY_INSUFFICIENT, ICONS, INTERNAL, INTERVAL_NOTE, NOT_RATED, UNADJUSTED, display_label

CHOOSE = "(choose)"
EXAMPLE = "mid-size Canadian SaaS, $80M revenue, 25% margins, in US and UK"
# Streamlit drops a widget's session_state key while its page is not rendered. Copies live under "saved_inputs" (not a widget key) and re-seed the widgets each run.
EDIT_KEYS = ("prompt", "edit_industry", "edit_home", "edit_existing", "edit_revenue", "edit_margin", "edit_model")


def _reseed():
    saved = st.session_state.setdefault("saved_inputs", {})
    for k, v in saved.items():
        st.session_state.setdefault(k, v)


def _save():
    st.session_state["saved_inputs"].update({k: st.session_state[k] for k in EDIT_KEYS if k in st.session_state})


def _parse():
    try:
        res = parse_prompt(st.session_state.get("prompt", ""), S.client())
    except Exception as e:  # network, validation, empty text
        st.session_state["parse_error"] = f"Could not parse the description: {e}"
        st.session_state["draft"] = None
        return
    st.session_state["parse_error"] = None
    st.session_state["draft"] = ProfileDraft(res.profile)
    st.session_state["result"] = None
    p = res.profile
    st.session_state.update({"edit_industry": p.industry or CHOOSE, "edit_home": p.home_country or "", "edit_existing": ", ".join(p.existing_markets),
                             "edit_revenue": None if p.revenue_usd is None else p.revenue_usd / 1e6,
                             "edit_margin": None if p.operating_margin is None else p.operating_margin * 100, "edit_model": p.business_model or ""})


def _open_detail(country):
    st.session_state["detail_country"] = country
    st.session_state["page"] = "Market detail"


def verdict_style(label):
    v = INTERNAL.get(label, label)
    c = S.VERDICT_COLORS.get(v, "#ffffff")
    return f"background-color: {c}; color: {'#ffffff' if v in ('expand', 'avoid') else '#000000'}; font-weight: 600"


def _cat_trace(locs, color, name, hover, text=None, **kw):
    """One Choropleth per category. A flat scale must span 0 to 1: stops both at 0 are invalid and Plotly falls back to its default blue."""
    return go.Choropleth(locations=locs, z=[0] * len(locs), zmin=0, zmax=1, colorscale=[[0, color], [1, color]], showscale=False, name=name, showlegend=True,
                         text=text, hovertemplate=hover, **kw)


def build_map(cand, unscored, exclude):
    iso = {c: m["iso3"] for c, m in S.countries().items()}
    fig = go.Figure()
    for v in ("expand", "consider", "avoid", "not_rated"):
        d = cand[cand.display_key == v]
        if v == "not_rated" and not len(d):
            continue
        name = NOT_RATED if v == "not_rated" else DISPLAY[v]
        fig.add_trace(_cat_trace([iso[c] for c in d.country], S.VERDICT_COLORS[v], name, "%{text}: " + name + "<extra></extra>", d.country_name, marker_line_color="white"))
    if len(unscored):
        locs = [iso[c] for c in unscored.country if c in iso]
        fig.add_trace(_cat_trace(locs, S.VERDICT_COLORS["insufficient_data"], DISPLAY_INSUFFICIENT, "%{location}: insufficient data<extra></extra>"))
        fig.add_trace(go.Scattergeo(locations=locs, locationmode="ISO-3", mode="markers", marker=dict(symbol="hash", size=11, color="#4a4a4a", line=dict(width=1, color="#4a4a4a")),
                                    showlegend=False, hoverinfo="skip"))
    ex = [iso[c] for c in exclude if c in iso]
    if ex:
        name = "Existing / home market (excluded)"
        fig.add_trace(_cat_trace(ex, "#ffffff", name, "%{location}: excluded<extra></extra>", marker_line_color="#444444", marker_line_width=1.5).update(showlegend=False))
        # the Choropleth legend swatch ignores its outline, so a legend-only marker carries the entry
        fig.add_trace(go.Scattergeo(lat=[None], lon=[None], mode="markers", name=name, marker=dict(symbol="square", size=11, color="#ffffff", line=dict(width=1.5, color="#444444"))))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=430, legend=dict(orientation="h", y=-0.05),
                      geo=dict(showframe=False, showcoastlines=False, showland=True, projection_type="natural earth", bgcolor="rgba(0,0,0,0)", landcolor="#eef0f2"))
    return fig


def results_table(cand):
    f = lambda x: "n/a" if pd.isna(x) else f"{x:.1%}"
    rk = lambda r, c: "n/a" if pd.isna(r[c]) else f"{int(r[c])} of {int(r.n_ranked_in_industry)}"
    t = pd.DataFrame({
        "Country": cand.country_name, "Verdict": cand.apply(display_label, axis=1), "Flags": cand.flag_icons,
        "Recent growth signal (listed peers)": cand.recent_growth_signal.map(f), "Growth rank": cand.apply(lambda r: rk(r, "recent_growth_rank"), axis=1),
        "Model estimate [80% interval]": cand.apply(lambda r: f"{r.model_estimate:.1%} [{r.model_estimate_lo:.1%} to {r.model_estimate_hi:.1%}]", axis=1),
        "Recent downside volatility": cand.recent_downside_signal.map(f), "Downside rank": cand.apply(lambda r: rk(r, "recent_downside_rank"), axis=1),
        "Downside [80% interval]": cand.apply(lambda r: "n/a" if pd.isna(r.recent_downside_estimate_lo) else f"{r.recent_downside_estimate_lo:.1%} to {r.recent_downside_estimate_hi:.1%}", axis=1),
        "Data freshness": cand.data_freshness.map(lambda x: "n/a" if pd.isna(x) else f"{x:.0%}")})
    return t


def profile_card(draft):
    p = draft.profile
    st.subheader("Parsed profile")
    if p.status == "unsupported":
        st.error("**Unsupported.** " + unsupported_message(p) + " No score is produced.")
        return False
    if p.status == "needs_info":
        st.warning(unsupported_message(p))
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Industry", [CHOOSE, *INDUSTRIES], key="edit_industry")
    c2.text_input("Home country (ISO code, removed from candidates)", key="edit_home")
    c3.text_input("Existing markets (ISO codes, comma separated, removed from candidates)", key="edit_existing")
    c4, c5, c6 = st.columns(3)
    c4.number_input("Revenue (USD millions)", min_value=0.0, value=None, key="edit_revenue")
    c5.number_input("Operating margin (%)", value=None, key="edit_margin")
    c6.text_input("Business model", key="edit_model")
    st.caption(f"Parser confidence: {p.confidence}. Missing: {', '.join(p.missing_fields()) or 'nothing'}. The profile only selects the industry and removes your home and existing markets; it is never a model input.")
    return True


def do_score():
    draft = st.session_state["draft"]
    ind = st.session_state["edit_industry"]
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    try:
        draft.edit(industry=None if ind == CHOOSE else ind, status="supported" if ind != CHOOSE else "needs_info", home_country=st.session_state["edit_home"] or None,
                   existing_markets=split(st.session_state["edit_existing"]),
                   revenue_usd=None if st.session_state["edit_revenue"] is None else st.session_state["edit_revenue"] * 1e6,
                   operating_margin=None if st.session_state["edit_margin"] is None else st.session_state["edit_margin"] / 100,
                   business_model=st.session_state["edit_model"] or None, unsupported_reason=None)
        draft.confirm()
    except (ValidationError, ValueError) as e:
        st.session_state["score_error"] = str(e)
        return
    st.session_state["score_error"] = None
    p = draft.profile
    exclude = set(p.existing_markets) | ({p.home_country} if p.home_country else set())
    cand, unscored = candidate_table(p.industry, exclude)
    st.session_state["result"] = {"profile": p, "cand": cand, "unscored": unscored, "exclude": exclude}
    st.session_state["narr"] = {}


def results(res):
    cand, unscored, exclude, p = res["cand"], res["unscored"], res["exclude"], res["profile"]
    st.subheader(f"Results: {p.industry.replace('_', ' ')}")
    counts = cand.display_key.value_counts()
    nr = int(counts.get("not_rated", 0))
    cols = st.columns([2, 1, 1, 1] + ([1] if nr else []))  # the first label is long
    for col, v in zip(cols, ("expand", "consider", "avoid")):
        col.metric(DISPLAY[v], int(counts.get(v, 0)))
    cols[3].metric(DISPLAY_INSUFFICIENT, len(unscored))
    if nr:
        cols[4].metric(NOT_RATED, nr)
    st.plotly_chart(build_map(cand, unscored, exclude), width="stretch", key="result_map")
    st.caption("Grey with hatch marks = insufficient data. Blue-grey = not rated (reference only). White with a dark outline = your home / existing market (excluded). " + ORDER_NOTE)
    t = results_table(cand)
    st.dataframe(t.style.map(verdict_style, subset=["Verdict"]), hide_index=True, width="stretch", key="result_table",
                 column_config={"Verdict": st.column_config.TextColumn("Verdict", width=300)})
    st.caption("Flags: " + "   ".join(f"{i} {d}" for i, d in ICONS.values()))
    st.caption("**Recent growth signal** = what listed peers actually did recently (drives rank and label). **Model estimate** = a ridge estimate shrunk toward the industry mean, with an 80% interval. They are different things. " + INTERVAL_NOTE
               + (" Growth for Turkey is unadjusted for inflation (not CPI-deflated)." if cand.country.isin(UNADJUSTED).any() else ""))
    if len(unscored):
        u = pd.DataFrame({"Country": unscored.country_name.fillna(unscored.country), "Status": "▨ insufficient data",
                          "Reason": [S.insufficient_reason(r) for _, r in unscored.iterrows()]})
        st.dataframe(u, hide_index=True, width="stretch", key="unscored_table")
    c1, c2 = st.columns([3, 1])
    pick = c1.selectbox("Open market detail for", cand.country.tolist(), format_func=lambda c: cand.set_index("country").country_name[c], key="pick_country")
    c2.button("Open detail", key="open_detail", on_click=_open_detail, args=(pick,))


def render():
    _reseed()
    st.header("Screen markets for a company")
    st.text_area("Describe the company (industry, home country, revenue, margin, markets you are already in)", key="prompt", height=100, placeholder=EXAMPLE)
    st.button("Parse description", key="parse_btn", on_click=_parse)
    if st.session_state.get("parse_error"):
        st.error(st.session_state["parse_error"])
    draft = st.session_state.get("draft")
    if draft is None:
        _save()
        return
    ok = profile_card(draft)
    _save()
    if not ok:
        return
    st.button("Score markets", key="score_btn", disabled=st.session_state.get("edit_industry", CHOOSE) == CHOOSE, on_click=do_score)
    if st.session_state.get("score_error"):
        st.error("Please fix the profile: " + st.session_state["score_error"])
    if st.session_state.get("result"):
        results(st.session_state["result"])

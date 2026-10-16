"""AppTest coverage of the dashboard: banner on every page, screen flow, each verdict path, insufficient data, unsupported/needs-info
profiles, the signal-outside-interval note, and the template-narration marker. All API calls are replaced by a fake client."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener.app import services as S
from screener.parse import parser
from screener.parse.narrate import LABEL_DEFINITIONS
from screener.score.labels import DISPLAY, INTERVAL_NOTE, NOT_RATED, TIER_NOTE

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")
SAAS = dict(industry="software", unsupported_reason=None, home_country="CA", existing_markets=["US", "GB"], revenue_usd=8e7, operating_margin=0.25,
            business_model="SaaS", confidence="high", unparsed_notes=[])
BANK = dict(SAAS, industry="unsupported", unsupported_reason="Banks are outside the scored industries.", home_country="US", existing_markets=[], revenue_usd=None, operating_margin=None)
VAGUE = dict(SAAS, industry="needs_info", home_country=None, existing_markets=[], revenue_usd=None, operating_margin=None, confidence="low")
NARR_OK = ("Recent growth of listed peers was [[recent_growth_signal]] (rank [[recent_growth_rank]]). The model estimate is [[model_estimate]] with an [[interval_level]] "
           "interval from [[model_estimate_low]] to [[model_estimate_high]]. This is a low-skill screen.")
NARR_BAD = "Growth was 9.9% so you should invest."


class FakeClient:
    def __init__(self, narration=NARR_OK):
        self.narration = narration
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        if "tools" in kw:  # parser
            text = kw["messages"][0]["content"].lower()
            tool = BANK if "bank" in text else VAGUE if "vague" in text else SAAS
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="record_profile", input=tool)])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.narration)])


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(parser, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(S, "client", lambda: FakeClient())

    def make(narration=NARR_OK):
        monkeypatch.setattr(S, "client", lambda: FakeClient(narration))
        return AppTest.from_file(APP, default_timeout=90).run()
    return make


def parse(at, text="saas company"):
    at.text_area(key="prompt").set_value(text)
    at.button(key="parse_btn").click().run()
    return at


def score(at):
    at.button(key="score_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def go(at, page):
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_banner_on_every_page(app):
    at = app()
    for page in ("Screen", "Market detail", "Label explainer", "Model card"):
        go(at, page)
        banner = at.warning[0].value
        assert "not causal advice or a forecast" in banner and "30 Sep 2026" in banner and "Data snapshot: " in banner and "shipped-v1.1" in banner and "Low-skill" in banner


def test_screen_flow_removes_existing_markets_and_orders_by_verdict(app):
    at = score(parse(app()))
    assert at.selectbox(key="edit_industry").value == "software" and at.text_input(key="edit_home").value == "CA"
    t = at.dataframe[0].value
    assert not {"Canada", "United States", "United Kingdom"} & set(t.Country) and len(t) > 15
    order = t.Verdict.map({v: i for i, v in enumerate([DISPLAY["expand"], DISPLAY["consider"], DISPLAY["avoid"], NOT_RATED])})
    assert order.is_monotonic_increasing and set(t.Verdict) == set(DISPLAY.values()) | {NOT_RATED}  # Turkey is shown as Not rated, last
    assert "Recent growth signal (listed peers)" in t.columns and "Model estimate [80% interval]" in t.columns
    assert len(at.get("plotly_chart")) >= 1


def test_insufficient_data_shown_with_reason(app):
    at = score(parse(app()))
    u = at.dataframe[1].value
    assert "New Zealand" in set(u.Country) and u.Reason.str.contains("at least 5").all()


def test_unsupported_profile_cannot_proceed(app):
    at = parse(app(), "regional bank in ohio")
    assert any("Unsupported" in e.value and "Banks are outside" in e.value and "distribution" in e.value for e in at.error)
    assert len([b for b in at.button if b.key == "score_btn"]) == 0


def test_needs_info_requires_choosing_an_industry(app):
    at = parse(app(), "vague tech thing")
    assert any("cannot tell which industry" in w.value for w in at.warning)
    assert at.button(key="score_btn").disabled
    at.selectbox(key="edit_industry").set_value("software").run()
    assert not at.button(key="score_btn").disabled


@pytest.mark.parametrize("verdict", ["expand", "consider", "avoid"])
def test_each_verdict_path_in_detail(app, verdict):
    at = score(parse(app()))
    cand = at.session_state["result"]["cand"]
    c = cand[(cand.verdict == verdict) & (~cand.low_confidence)].iloc[0].country
    go(at, "Market detail")
    at.selectbox(key="detail_country").set_value(c).run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert LABEL_DEFINITIONS[verdict] in md and DISPLAY[verdict] in md
    labels = [m.label for m in at.metric]
    assert "Model estimate of growth" in labels and any(l.startswith("Recent growth signal") for l in labels)  # separate labelled fields


def test_comparable_firms_are_names_only(app):
    f = S.comparable_firms("ES", "software", 2025)
    assert list(f.columns) == ["Firm", "Ticker"] and len(f) >= 5 and f.Firm.notna().all()
    assert list(pd.read_csv(S.vp("data/panel/comparable_firms.csv")).columns) == ["country", "industry", "as_of_year", "name", "ticker"]


def test_turkey_is_flagged_never_actionable(app):
    at = score(parse(app()))
    go(at, "Market detail")
    at.selectbox(key="detail_country").set_value("TR").run()
    assert any("reference only" in w.value and "unadjusted for inflation" in w.value for w in at.warning)
    md = " ".join(m.value for m in at.markdown)
    assert NOT_RATED in md and DISPLAY["consider"] not in md  # internal tier stays 'consider', but the label is never shown for TR
    assert "unadjusted for inflation (not CPI-deflated)" in " ".join(m.label for m in at.metric)


def test_signal_outside_interval_note(app):
    at = score(parse(app()))
    cand = at.session_state["result"]["cand"]
    c = cand[cand.signal_outside_interval == True].iloc[0].country
    go(at, "Market detail")
    at.selectbox(key="detail_country").set_value(c).run()
    assert any("the model expects partial reversion" in i.value for i in at.info)


def test_llm_narration_then_template_marker(app):
    at = score(parse(app(NARR_OK)))
    go(at, "Market detail")
    at.button(key="narrate_btn").click().run()
    caps = " ".join(c.value for c in at.caption)
    assert "Written by a language model" in caps and "Template narration" not in caps
    at = score(parse(app(NARR_BAD)))
    go(at, "Market detail")
    at.button(key="narrate_btn").click().run()
    assert "Template narration" in " ".join(c.value for c in at.caption)
    assert "[[" not in " ".join(m.value for m in at.markdown)


def test_explainer_and_model_card_content(app):
    at = go(app(), "Label explainer")
    txt = " ".join(m.value for m in at.markdown)
    assert "+0.041" in txt and "-0.017" in txt and "percentage points" in txt
    for d in LABEL_DEFINITIONS.values():
        assert d in txt
    assert TIER_NOTE in " ".join(i.value for i in at.info)
    m = at.dataframe[0].value  # display <-> internal mapping table
    assert list(m["Display label"]) == [*DISPLAY.values(), NOT_RATED] and list(m["Internal tier name"]) == [*DISPLAY, "consider"]
    assert "confidence interval" not in txt.lower() and "CI" not in " ".join(c for df in at.dataframe for c in df.value.columns)
    at = go(at, "Model card")
    assert at.subheader[0].value.startswith("1. The pre-registered protocol")
    assert len(at.dataframe) >= 3
    assert "confidence interval" not in " ".join(m.value for m in at.markdown).lower()
    cards = [df.value for df in at.dataframe if "Display label" in df.value.columns]
    assert len(cards) == 1 and list(cards[0]["Internal tier name"]) == [*DISPLAY, "consider"]
    assert "CI" not in " ".join(c for df in at.dataframe for c in df.value.columns)


def test_map_colours_per_category():
    """Regression: every country used to render in Plotly's default blue because the flat colourscales were invalid."""
    from screener.app.views.screen import build_map
    from screener.score.candidates import candidate_table
    exclude = {"CA", "US", "GB"}
    cand, unscored = candidate_table("software", exclude)
    assert len(unscored), "need an insufficient_data cell to check grey"
    fig = build_map(cand, unscored, exclude)
    iso = {c: m["iso3"] for c, m in S.countries().items()}
    colour = {}
    for t in fig.data:
        if t.type != "choropleth":
            continue
        scale = [c for _, c in t.colorscale]
        assert [p for p, _ in t.colorscale] == [0, 1] and len(set(scale)) == 1  # valid, flat scale
        colour.update({loc: (scale[0], t.marker.line.color) for loc in t.locations})
    for v in ("expand", "consider", "avoid", "not_rated"):  # Turkey is its own category, not Neutral
        assert {colour[iso[c]][0] for c in cand[cand.display_key == v].country} == {S.VERDICT_COLORS[v]}
    assert len({S.VERDICT_COLORS[v] for v in ("expand", "consider", "avoid", "not_rated", "insufficient_data")}) == 5
    assert {colour[iso[c]][0] for c in unscored.country if c in iso} == {"#9aa0a6"}
    for c in exclude:
        fill, outline = colour[iso[c]]
        assert fill == "#ffffff" and outline not in (None, "white", "#ffffff")


def test_profile_card_survives_page_switch(app):
    """Regression: the parsed-profile card (and the prompt) emptied after visiting another page and coming back."""
    at = parse(app(), "mid-size Canadian SaaS, $80M revenue, 25% margins, in US and UK")
    at.number_input(key="edit_margin").set_value(30.0).run()  # a user edit must survive too
    want = {k: at.session_state[k] for k in ("edit_industry", "edit_home", "edit_existing", "edit_revenue", "edit_margin", "edit_model")}
    assert want["edit_industry"] == "software" and want["edit_home"] == "CA" and want["edit_margin"] == 30.0
    for page in ("Label explainer", "Model card"):
        at.sidebar.radio(key="page").set_value(page).run()
    at.sidebar.radio(key="page").set_value("Screen").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.text_area(key="prompt").value.startswith("mid-size Canadian SaaS")
    assert at.selectbox(key="edit_industry").value == "software"
    assert at.text_input(key="edit_home").value == "CA" and at.text_input(key="edit_existing").value == "US, GB"
    assert at.number_input(key="edit_revenue").value == 80.0 and at.number_input(key="edit_margin").value == 30.0
    assert at.text_input(key="edit_model").value == "SaaS"


def test_freshness_metric_says_late_reporters_when_flagged(app):
    at = score(parse(app()))
    go(at, "Market detail")
    at.selectbox(key="detail_country").set_value("AU").run()  # 70% of firms reported, with a late-reporters flag
    assert any("Late reporters" in m.value for m in at.markdown)
    fresh = [m for m in at.metric if m.label.startswith("Firms with the latest year")][0]
    assert fresh.delta == "late reporters"
    assert INTERVAL_NOTE in " ".join(c.value for c in at.caption)

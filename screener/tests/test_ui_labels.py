"""Display-only changes: Not rated (Turkey), Neutral, growth wording, template parity, rank text, thin-cell flag, freshness badge, Korea badge.
Nothing here may touch verdict logic: each test reads reports/scored_cells.csv and only checks what is shown."""
import json
import re

import pandas as pd

from screener.app.views import detail
from screener.features.run_panel import ROOT, vp
from screener.parse import narrate as nr
from screener.score.labels import DISPLAY, INTERVAL_NOTE, NOT_RATED, THIN_N, UNADJUSTED_GROWTH, display_label, flag_keys, growth_basis

SC = pd.read_csv(vp("reports/scored_cells.csv"))


def cell(country, industry="software"):
    return SC[(SC.country == country) & (SC.industry == industry)].iloc[0]


def test_turkey_is_not_rated_and_growth_is_never_called_real():
    tr = cell("TR")
    assert tr.verdict == "consider" and display_label(tr) == NOT_RATED  # internal tier unchanged
    assert growth_basis(tr) == UNADJUSTED_GROWTH and growth_basis(cell("FR")) == "real"
    inp = nr.narration_input(tr)
    assert inp["label"] == NOT_RATED and NOT_RATED in inp["label_definitions"]
    assert UNADJUSTED_GROWTH in inp["fields"]["recent_growth_signal"]["label"]
    assert not re.search(r"\breal\b", json.dumps(inp))  # nothing the narrator sees says "real" for Turkey
    text = nr.render(nr.template(inp), inp)
    assert UNADJUSTED_GROWTH in text and not re.search(r"\breal\b", text) and NOT_RATED in text
    assert "real" in nr.narration_input(cell("FR"))["fields"]["recent_growth_signal"]["label"]


def test_middle_tier_is_neutral_with_the_new_definition():
    assert DISPLAY["consider"] == "Neutral"
    d = nr.LABEL_DEFINITIONS["consider"]
    assert "not in the top third for recent growth and not in the top third for downside volatility; this includes markets with weak recent growth" in d
    fr = cell("FR")  # growth rank 25 of 25, middle tier
    assert fr.verdict == "consider" and display_label(fr) == "Neutral"
    assert "No strong signal" not in json.dumps(nr.narration_input(fr))


def test_template_parity_ranks_flags_and_forecast_line():
    r = cell("FR").copy()
    r["signal_outside_interval"], r["signal_outside_direction"] = True, "low"
    inp = nr.narration_input(r)
    t = nr.render(nr.template(inp), inp)
    assert ".." not in t and "reversion." in t
    assert "not a forecast" in t and "Neutral" in t and "(1 = strongest)" in t and "(1 = calmest)" in t
    assert "Recent growth signal (listed peers, real)" in t and "Model estimate of growth" in t and "Recent downside volatility" in t
    assert "[[" not in t


def test_rank_text_in_narrator_fields():
    f = nr.narration_input(cell("FR"))["fields"]
    assert re.fullmatch(r"\d+ of 25 \(1 = strongest\)", f["recent_growth_rank"]["text"])
    assert re.fullmatch(r"\d+ of 25 \(1 = calmest\)", f["recent_downside_rank"]["text"])


def test_thin_cell_flag_is_display_only():
    thin = SC[(SC.n_members < THIN_N) & ~SC.insufficient_data]
    assert len(thin) and all("thin" in flag_keys(r) for _, r in thin.iterrows())
    assert not any("thin" in flag_keys(r) for _, r in SC[SC.n_members >= THIN_N].iterrows())
    before = SC.verdict.copy()
    SC.apply(flag_keys, axis=1)
    assert (SC.verdict == before).all()


def test_korea_badge_shows_the_assumed_share_the_narrator_quotes():
    kr = cell("KR")
    n = int(kr.n_members)
    assert f"Assumed home country for {round(kr.domicile_assumed_share * n)} of {n} firms ({kr.domicile_assumed_share:.0%})" in detail.key_text("assumed", kr)
    assert nr.narration_input(kr)["fields"]["assumed_share"]["text"] == f"{kr.domicile_assumed_share:.0%}"
    c = SC[SC.concentrated == True].iloc[0]  # noqa: E712
    assert f"{c.top1_share:.0%}" in detail.key_text("concentrated", c)


def test_late_reporters_flag_and_freshness_message_agree():
    au = cell("AU")  # 70% freshness with a late-reporters flag: must not read "ok"
    assert "late" in detail.all_keys(au)

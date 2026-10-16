"""Offline tests of the parser, confirmation step, candidate table and narrator post-check (no API calls: a fake client)."""
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from screener.parse import narrate as nr
from screener.parse import parser
from screener.parse.schema import CompanyProfile
from screener.score.candidates import candidate_table


class FakeMessages:
    def __init__(self, replies):
        self.replies, self.calls, self.kwargs = list(replies), 0, []

    def create(self, **kw):
        self.calls += 1
        self.kwargs.append(kw)
        r = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        if isinstance(r, dict):
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="record_profile", input=r)])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=r)])


class Fake:
    def __init__(self, *replies):
        self.messages = FakeMessages(replies)


SAAS = dict(industry="software", unsupported_reason=None, home_country="ca", existing_markets=["us", "GB", "ca"], revenue_usd=8e7,
            operating_margin=0.25, business_model="SaaS", confidence="high", unparsed_notes=[])
BANK = dict(SAAS, industry="unsupported", unsupported_reason="Banks are outside the scored industries.", home_country="US", existing_markets=[], revenue_usd=None, operating_margin=None)


def test_forced_tool_call_temperature_zero_and_normalisation(tmp_path, monkeypatch):
    monkeypatch.setattr(parser, "CACHE_DIR", tmp_path)
    f = Fake(SAAS)
    r = parser.parse_prompt("canadian saas", f)
    kw = f.messages.kwargs[0]
    assert kw["tool_choice"] == {"type": "tool", "name": "record_profile"} and kw["extra_body"] == {"temperature": 0}
    p = r.profile
    assert (p.status, p.industry, p.home_country) == ("supported", "software", "CA")
    assert p.existing_markets == ["US", "GB"]  # uppercased, deduped, home market removed


def test_parse_cache_hits_without_api(tmp_path, monkeypatch):
    monkeypatch.setattr(parser, "CACHE_DIR", tmp_path)
    f = Fake(SAAS)
    parser.parse_prompt("x", f)
    r2 = parser.parse_prompt("x", Fake())  # a client that would fail on any call
    assert r2.cached and f.messages.calls == 1


def test_unsupported_and_needs_info_paths():
    bank = parser.to_profile(BANK)
    assert bank.status == "unsupported" and bank.industry is None
    vague = parser.to_profile(dict(SAAS, industry="needs_info", home_country=None, existing_markets=[], revenue_usd=None, operating_margin=None, confidence="low"))
    assert vague.status == "needs_info" and "industry" in vague.missing_fields()
    with pytest.raises(ValueError):
        parser.ProfileDraft(bank).confirm()  # cannot score an unsupported profile


def test_schema_rejects_bad_values():
    for bad in (dict(home_country="Canada"), dict(operating_margin=25), dict(revenue_usd=-1), dict(existing_markets=["USA"])):
        with pytest.raises(ValidationError):
            CompanyProfile(**{**dict(status="supported", industry="software"), **bad})
    with pytest.raises(ValidationError):
        CompanyProfile(status="supported", industry=None)
    with pytest.raises(ValidationError):
        CompanyProfile(status="unsupported")


def test_edit_revalidates_and_resets_confirmation():
    d = parser.ProfileDraft(parser.to_profile(SAAS)).confirm()
    d.edit(industry="retail", operating_margin=0.1)
    assert d.profile.industry == "retail" and not d.confirmed
    with pytest.raises(ValidationError):
        d.edit(operating_margin=40)


def test_candidates_remove_existing_and_home_markets():
    cand, unscored = candidate_table("software", {"CA", "US", "GB"})
    assert not {"CA", "US", "GB"} & set(cand.country) and set(cand.verdict) <= {"expand", "consider", "avoid"}
    assert (cand.industry == "software").all() and (unscored.country == "NZ").any()
    # ordering: verdict tier, then unflagged before flagged
    tier = cand.display_key.map({"expand": 0, "consider": 1, "avoid": 2, "not_rated": 3})  # Turkey (internal 'consider') is displayed last as Not rated
    assert tier.is_monotonic_increasing
    for t in (0, 1, 2):
        assert (cand[tier == t].n_flags > 0).astype(int).is_monotonic_increasing


def row(**kw):
    base = dict(country="XX", country_name="Testland", industry="software", verdict="expand", scored_asof_year=2025, n_members=20,
                recent_growth_signal=0.081, recent_growth_rank=4, n_ranked_in_industry=26, model_estimate=0.053, model_estimate_lo=0.004,
                model_estimate_hi=0.102, recent_downside_signal=0.142, recent_downside_rank=9, recent_downside_estimate_lo=0.108,
                recent_downside_estimate_hi=0.176, low_confidence=False, concentrated=False, top1_share=0.3, domicile_assumed_share=0.0,
                interval_reliable=True, lower_reliability=False, drop_flag=False, signal_outside_interval=False, signal_outside_direction="")
    return pd.Series({**base, **kw})


GOOD = ("Testland is labelled Strong momentum, volatility not elevated. Recent growth was [[recent_growth_signal]], rank [[recent_growth_rank]]. The model estimate is [[model_estimate]], "
        "with an [[interval_level]] interval from [[model_estimate_low]] to [[model_estimate_high]]. This is a low-skill screen.")


def test_postcheck_enforces_labelled_fields_and_blocks_free_numbers():
    inp = nr.narration_input(row())
    assert nr.check(GOOD, inp) == []
    assert nr.render(GOOD, inp).count("8.1%") == 1 and "5.3%" in nr.render(GOOD, inp)
    assert any("free-typed" in b for b in nr.check("Growth was 9.3%.", inp))                                    # typed number
    assert any("free-typed" in b for b in nr.check("Growth was 8.1%.", inp))                                    # even a CORRECT number must be a placeholder
    assert any("unknown placeholder" in b for b in nr.check("Growth was [[growth_made_up]].", inp))
    assert any("banned" in b for b in nr.check("You should invest here.", inp))
    assert any("banned" in b for b in nr.check("Growth is driven by GDP.", inp))
    assert any("banned" in b for b in nr.check("The 80% confidence interval is wide.", inp))   # use "prediction interval"
    assert nr.check("The 80% prediction interval runs from [[model_estimate_low]] to [[model_estimate_high]].", inp) == []
    assert any("banned" in b for b in nr.check("Avoid this market.", inp))
    assert any("volatility sentence" in b for b in nr.check("Volatility was [[recent_growth_signal]].", inp))   # growth number in a volatility sentence
    assert any("growth sentence" in b for b in nr.check("Growth was [[recent_downside_signal]].", inp))        # downside number in a growth sentence
    assert any("without saying estimate" in b for b in nr.check("Growth reached [[model_estimate_high]].", inp))
    assert nr.check(nr.template(inp), inp) == []  # the fallback template itself passes the check


def test_render_absorbs_stray_percent_sign():
    inp = nr.narration_input(row())
    assert nr.render("Growth was [[recent_growth_signal]]% and rank [[recent_growth_rank]].", inp) == "Growth was 8.1% and rank 4 of 26 (1 = strongest)."
    assert "%%" not in nr.render("Estimate [[model_estimate]]% , level [[interval_level]]%.", inp)


def test_narrator_retries_then_falls_back_to_template_and_marks_it():
    n = nr.narrate(row(), Fake("Growth is 9.9% so you should invest.", GOOD))
    assert n.source == "llm" and "8.1%" in n.text and "[[" not in n.text
    n = nr.narrate(row(), Fake("Growth is 9.9%.", "Growth is 7.7%."))
    assert n.source == "template" and n.violations and "[[" not in n.text
    sent = nr.narration_input(row())
    assert set(sent) == {"country", "industry", "label", "label_definitions", "growth_basis", "fields", "flags", "caveats"}
    assert sent["label"] == "Strong momentum, volatility not elevated"
    assert sent["label_definitions"]["Elevated volatility"] == "Elevated volatility = higher recent market volatility than industry peers."
    assert sent["fields"]["recent_growth_signal"]["text"] != sent["fields"]["model_estimate"]["text"]  # separate labelled fields


def test_flags_reach_narrator_input_including_outside_interval_note():
    inp = nr.narration_input(row(low_confidence=True, concentrated=True, top1_share=0.7, domicile_assumed_share=0.4, lower_reliability=True, drop_flag=True,
                                 signal_outside_interval=True, signal_outside_direction="high"))
    assert len(inp["flags"]) == 6 and any("never given an actionable label" in f for f in inp["flags"])
    assert "Recent growth is unusually high versus the model estimate; the model expects partial reversion." in inp["flags"]
    assert nr.check(nr.template(inp), inp) == [] and "70%" in nr.render(nr.template(inp), inp)


def test_unsupported_message_has_reason_and_scope():
    msg = parser.unsupported_message(parser.to_profile(BANK))
    assert "Banks are outside" in msg and "distribution" in msg and "pre-revenue" in msg


def test_missing_api_key_gives_a_clear_message(monkeypatch):
    import dotenv
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match=".env.example"):
        parser.get_client()

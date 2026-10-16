"""Narrator: describes ONE scored row in plain language. Input is only the scored row (pre-formatted), flags, caveats and the label
definitions as JSON. It may describe only the persistence signals, intervals and caveats - no macro-driver claims, no advice language.

Numbers: the narrator never types a number. It writes [[field]] placeholders; the code substitutes each placeholder with the display
string of THAT labelled field, so every number matches its field by construction. The post-check enforces this: unknown placeholders,
any free-typed number outside the static caveat/level numbers, banned wording, and sentence-level label consistency (growth tokens
never in a volatility sentence and vice versa; estimate/interval tokens only in sentences that say estimate/interval/model).
Failures retry once, then a deterministic template is used and the result is marked source="template"."""
import json
import re
from dataclasses import dataclass

import pandas as pd

from screener.parse.parser import MODEL, get_client
from screener.score.labels import DISPLAY, NOT_RATED, display_label, growth_basis, outside_note

LABEL_DEFINITIONS = {  # keyed by internal tier name; the text uses the display label
    "avoid": f"{DISPLAY['avoid']} = higher recent market volatility than industry peers.",
    "expand": f"{DISPLAY['expand']} = stronger recent growth among industry peers without elevated volatility.",
    "consider": f"{DISPLAY['consider']} = not in the top third for recent growth and not in the top third for downside volatility; this includes markets with weak recent growth.",
    "not_rated": f"{NOT_RATED} = scored for reference only; the country is excluded from the model's training data, so it gets no label.",
}
DEFINITION_BY_LABEL = {**{DISPLAY[k]: LABEL_DEFINITIONS[k] for k in DISPLAY}, NOT_RATED: LABEL_DEFINITIONS["not_rated"]}
INDUSTRY_NAMES = {"non_bank_financials": "non-bank financial services", "software": "software and IT", "retail": "retail and e-commerce",
                  "healthcare": "healthcare and pharma"}
CAVEATS = [
    "This is a low-skill screen: out-of-sample rank correlation was about 0.53 for growth and 0.67 for downside volatility, measured on 2023-2024 data only.",
    "Labels describe recent momentum and volatility as of 30 Sep 2026, not a forecast for a new year.",
    "Growth is the global revenue growth of listed firms based in this country, not demand in this market.",
]
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
TOKEN = re.compile(r"\[\[([a-z_]+)\]\]%?")  # a stray % after a placeholder is absorbed in render() (values carry their own unit)
BANNED = re.compile(r"\b(should|recommend\w*|advis\w*|invest\w*|buy|sell|attractive|opportunit\w*|guarantee\w*|best|worst|gdp|macro\w*|econom\w*|"
                    r"inflation|interest rates?|population|driven by|due to|caused by|confidence intervals?|avoid|expand\w*)\b", re.I)
GROWTH_T = {"recent_growth_signal", "recent_growth_rank", "model_estimate", "model_estimate_low", "model_estimate_high"}
DOWN_T = {"recent_downside_signal", "recent_downside_rank", "downside_interval_low", "downside_interval_high"}
EST_T = {"model_estimate", "model_estimate_low", "model_estimate_high", "downside_interval_low", "downside_interval_high"}


def pct(x, d=1):
    return None if pd.isna(x) else f"{x * 100:.{d}f}%"


def _b(x):
    return bool(x) if pd.notna(x) else False


def narration_input(r) -> dict:
    n = int(r.n_ranked_in_industry) if pd.notna(r.n_ranked_in_industry) else None
    rank = lambda x, note: f"{int(x)} of {n} ({note})" if pd.notna(x) and n else "not ranked (low-confidence country)"
    basis = growth_basis(r)
    fields = {
        "recent_growth_signal": {"label": f"recent revenue growth of listed peers, {basis} (trailing; used for rank and label)", "text": pct(r.recent_growth_signal)},
        "recent_growth_rank": {"label": "rank of that growth among this industry's cells (1 = fastest)", "text": rank(r.recent_growth_rank, "1 = strongest")},
        "model_estimate": {"label": "model estimate of growth (ridge, shrunk toward the industry mean)", "text": pct(r.model_estimate)},
        "model_estimate_low": {"label": "low end of the 80% interval around the model estimate", "text": pct(r.model_estimate_lo)},
        "model_estimate_high": {"label": "high end of the 80% interval around the model estimate", "text": pct(r.model_estimate_hi)},
        "interval_level": {"label": "interval coverage level", "text": "80%"},
        "listed_firms": {"label": "listed firms in the cell", "text": str(int(r.n_members))},
        "cells_in_industry": {"label": "number of ranked cells in this industry", "text": str(n) if n else "n/a"},
        "as_of_year": {"label": "as-of fiscal year", "text": str(int(r.scored_asof_year))},
    }
    if pd.notna(r.recent_downside_signal):
        fields.update({
            "recent_downside_signal": {"label": "recent downside deviation of listed peers (trailing; also the estimate)", "text": pct(r.recent_downside_signal)},
            "recent_downside_rank": {"label": "rank of that volatility among this industry's cells (1 = calmest)", "text": rank(r.recent_downside_rank, "1 = calmest")},
            "downside_interval_low": {"label": "low end of the 80% interval around the downside deviation", "text": pct(r.recent_downside_estimate_lo)},
            "downside_interval_high": {"label": "high end of the 80% interval around the downside deviation", "text": pct(r.recent_downside_estimate_hi)}})
    flags = []
    if _b(r.low_confidence):
        flags.append("low-confidence country: scored for reference, never given an actionable label")
    if _b(r.concentrated):
        fields["concentration_share"] = {"label": "share of the cell's revenue from its largest listed firm", "text": pct(r.top1_share, 0)}
        flags.append("concentrated: the largest listed firm is [[concentration_share]] of the cell's revenue")
    if pd.notna(r.domicile_assumed_share) and r.domicile_assumed_share > 0:
        fields["assumed_share"] = {"label": "share of listed firms with an assumed home country", "text": pct(r.domicile_assumed_share, 0)}
        flags.append("[[assumed_share]] of listed firms have an assumed home country")
    if not _b(r.interval_reliable):
        flags.append("growth intervals unreliable for this industry")
    if _b(r.lower_reliability):
        flags.append("lower reliability industry: widest intervals and fewest scored cells")
    if _b(r.drop_flag):
        flags.append("20% or more of last year's firms had not yet reported the latest fiscal year")
    if _b(r.signal_outside_interval):
        flags.append(outside_note(r))
    return {"country": r.country_name, "industry": INDUSTRY_NAMES[r.industry], "label": display_label(r), "label_definitions": DEFINITION_BY_LABEL, "growth_basis": basis,
            "fields": fields, "flags": flags, "caveats": CAVEATS}


def static_numbers(inp: dict) -> set[float]:
    """Numbers that may be typed literally: those inside the static caveat/flag/definition sentences and the interval level."""
    txt = json.dumps({"c": inp["caveats"], "f": [f for f in inp["flags"]], "d": inp["label_definitions"], "lv": inp["fields"]["interval_level"]["text"]})
    return {round(float(m.replace(",", "")), 6) for m in NUM.findall(txt)}


def render(text: str, inp: dict) -> str:
    def val(m):
        v = inp["fields"][m.group(1)]["text"]
        return v + "%" if m.group(0).endswith("%") and not v.endswith("%") else v
    return TOKEN.sub(val, text)


def check(text: str, inp: dict) -> list[str]:
    bad = []
    fields = inp["fields"]
    for t in TOKEN.findall(text):
        if t not in fields:
            bad.append(f"unknown placeholder [[{t}]]")
    free = TOKEN.sub(" ", text)
    ok = static_numbers(inp)
    bad += [f"free-typed number {m!r} (use a [[placeholder]])" for m in NUM.findall(free) if round(float(m.replace(",", "")), 6) not in ok]
    bad += [f"banned wording {m.group(0)!r}" for m in BANNED.finditer(text)]
    for sent in re.split(r"(?<=[.!?])\s+", free if False else text):
        toks = set(TOKEN.findall(sent))
        low = sent.lower()
        if toks & GROWTH_T and re.search(r"\b(volatil\w*|downside)\b", low) and not toks & DOWN_T:
            bad.append(f"growth field in a volatility sentence: {sent[:60]!r}")
        if toks & DOWN_T and re.search(r"\bgrowth\b", low) and not toks & GROWTH_T:
            bad.append(f"downside field in a growth sentence: {sent[:60]!r}")
        if toks & EST_T and not re.search(r"\b(estimates?|intervals?|ranges?|ranging|band|between|uncertain\w*|model)\b", low):
            bad.append(f"estimate/interval field without saying estimate/interval: {sent[:60]!r}")
    return bad


SYSTEM = """You write a short plain-language description (3-5 sentences) of one screening result for a country-industry cell, for a business reader.
You receive JSON: the label, the label definitions, `fields` (each has a `label` saying exactly what it is, and a `text` value), flags and caveats.
NUMBERS: never type a digit yourself. Whenever you mention a number, write its placeholder exactly as [[field_name]] using the right field (for example [[recent_growth_signal]], [[model_estimate]], [[model_estimate_low]], [[model_estimate_high]], [[interval_level]]); the system fills in the value. Keep two things separate: the RECENT GROWTH SIGNAL (what actually happened recently; it drives the rank and label) and the MODEL ESTIMATE (an estimate with an interval). Never call the signal an estimate or the estimate a signal.
Rules: describe ONLY (1) the label, named exactly as given in `label`, using its definition, (2) the recent signals and their ranks, (3) the model estimate and the 80% prediction intervals (call them intervals or prediction intervals, never confidence intervals) and what their width means (wide means a rough guide), (4) the flags and caveats present in the JSON (copy the caveat and flag sentences faithfully; flags that contain [[placeholders]] keep them). Do not explain WHY the numbers are what they are: no claims about the economy, macro conditions, regulation, demand or causes, and never use 'driven by', 'due to' or 'caused by'; the only reason you may give for a label is its definition. No advice language (no 'should', 'recommend', 'invest', 'attractive', 'opportunity'). Never compute, convert or round numbers. If a flag is present, mention it. Always include one caveat sentence. Output the description only."""


@dataclass
class Narration:
    text: str
    source: str  # "llm" | "template"
    violations: list[str]
    raw: str = ""


def template(inp: dict) -> str:
    """Deterministic fallback. Same display labels and field names as the UI (see views/detail.py); never calls TR growth 'real'."""
    f = inp["fields"]
    s = [f"{inp['country']} ({inp['industry']}) is labelled {inp['label']}. {inp['label_definitions'][inp['label']]}",
         f"Recent growth signal (listed peers, {inp['growth_basis']}): [[recent_growth_signal]], rank [[recent_growth_rank]]. "
         "Model estimate of growth: [[model_estimate]], with an [[interval_level]] interval from [[model_estimate_low]] to [[model_estimate_high]]."]
    if "recent_downside_signal" in f:
        s.append("Recent downside volatility: [[recent_downside_signal]], rank [[recent_downside_rank]], with an [[interval_level]] "
                 "interval from [[downside_interval_low]] to [[downside_interval_high]].")
    s += [f"Flag: {x.rstrip('.')}." for x in inp["flags"]]
    s += [inp["caveats"][0], inp["caveats"][1]]  # the second one is the "not a forecast" line
    return " ".join(s)


def narrate(row, client=None) -> Narration:
    inp = narration_input(row)
    client = client or get_client()
    feedback, bad = "", []
    for _ in range(2):
        resp = client.messages.create(model=MODEL, max_tokens=600, extra_body={"temperature": 0}, system=SYSTEM + feedback,
                                      messages=[{"role": "user", "content": json.dumps(inp)}])
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        bad = check(raw, inp)
        if not bad:
            return Narration(render(raw, inp), "llm", [], raw)
        feedback = f"\n\nYour previous attempt was rejected: {'; '.join(bad)}. Rewrite it, using [[placeholders]] for every number."
    t = template(inp)
    return Narration(render(t, inp), "template", bad, t)

"""CLI: prompt -> parsed profile (editable, confirmed) -> scored candidate table (existing markets removed) -> narrated top/bottom cells.
python -m screener.cli "mid-size Canadian SaaS, $80M revenue, 25% margins, in US and UK" --yes [--set industry=software] [--top 3]"""
import argparse
import sys

import pandas as pd
import yaml

from screener.features.run_panel import ROOT
from screener.parse.narrate import narrate
from screener.parse.parser import ProfileDraft, get_client, parse_prompt, unsupported_message
from screener.score.candidates import ORDER_NOTE, candidate_table
from screener.score.labels import ICONS, INTERVAL_NOTE, display_label, outside_note

NAMES = {c: m["name"] for c, m in yaml.safe_load(open(ROOT / "screener/config/countries.yaml"))["countries"].items()}


def fmt_profile(p):
    rev = "n/a" if p.revenue_usd is None else f"${p.revenue_usd / 1e6:,.0f}M"
    mar = "n/a" if p.operating_margin is None else f"{p.operating_margin:.0%}"
    return (f"  status: {p.status}   industry: {p.industry}   confidence: {p.confidence}\n  home country: {p.home_country}   existing markets: {p.existing_markets}\n"
            f"  revenue: {rev} ({p.size_band})   operating margin: {mar}   business model: {p.business_model}\n"
            + (f"  unsupported reason: {p.unsupported_reason}\n" if p.unsupported_reason else "")
            + (f"  not mapped: {p.unparsed_notes}\n" if p.unparsed_notes else "") + f"  missing: {p.missing_fields()}")


def run(prompt, sets=None, top=3, narrate_cells=True, client=None, out=print):
    client = client or get_client()
    res = parse_prompt(prompt, client)
    draft = ProfileDraft(res.profile)
    out(f"PROMPT: {prompt}\n\nPARSED PROFILE ({'cache' if res.cached else 'API'}):\n{fmt_profile(draft.profile)}")
    for kv in sets or []:
        k, v = kv.split("=", 1)
        draft.edit(**{k: (v if k in ("industry", "home_country", "business_model") else float(v) if k in ("revenue_usd", "operating_margin") else v.split(","))})
        out(f"\nEDITED {k} -> {v}\n{fmt_profile(draft.profile)}")
    p = draft.profile
    if p.status != "supported":
        out(("\nUNSUPPORTED: " if p.status == "unsupported" else "\nNEEDS INFO: ") + unsupported_message(p) + " No score produced.")
        return None
    draft.confirm()
    exclude = set(p.existing_markets) | ({p.home_country} if p.home_country else set())
    cand, unscored = candidate_table(p.industry, exclude)
    unknown = sorted(c for c in exclude if c not in NAMES)
    out(f"\nCANDIDATES for {p.industry} (removed existing/home markets: {sorted(exclude) or 'none'}"
        + (f"; not in the scored set, nothing to remove: {unknown}" if unknown else "") + f")\n{ORDER_NOTE}")
    f1, f0 = (lambda x: "n/a" if pd.isna(x) else f"{x:.1%}"), (lambda r, a, b: "n/a" if pd.isna(r[a]) else f"{r[a]:.1%}..{r[b]:.1%}")
    rk = lambda r, c: "n/a" if pd.isna(r[c]) else f"{int(r[c])}/{int(r.n_ranked_in_industry)}"
    t = pd.DataFrame({"country": cand.country, "name": cand.country_name, "label": cand.apply(display_label, axis=1), "flags": cand.flag_icons,
                      "RECENT growth signal": cand.recent_growth_signal.map(f1), "g rank": cand.apply(lambda r: rk(r, "recent_growth_rank"), axis=1),
                      "MODEL estimate [80%]": cand.apply(lambda r: f"{r.model_estimate:.1%} [{r.model_estimate_lo:.1%}..{r.model_estimate_hi:.1%}]", axis=1),
                      "RECENT downside": cand.recent_downside_signal.map(f1), "d rank": cand.apply(lambda r: rk(r, "recent_downside_rank"), axis=1),
                      "downside [80%]": cand.apply(lambda r: f0(r, "recent_downside_estimate_lo", "recent_downside_estimate_hi"), axis=1)})
    out(t.to_string(index=False))
    out("flags: " + "  ".join(f"{i} {d}" for i, d in ICONS.values()))
    out(INTERVAL_NOTE)
    for _, r in cand[cand.signal_outside_interval == True].iterrows():
        out(f"  {r.country_name}: {outside_note(r)}")
    if len(unscored):
        out(f"Not scoreable (insufficient data): {', '.join(unscored.country_name.fillna(unscored.country))}")
    if narrate_cells and len(cand):
        picks = list(cand.head(top).index) + [i for i in cand.tail(top).index if i not in cand.head(top).index]
        out("\nNARRATED (top and bottom of the display order):")
        for i in picks:
            n = narrate(cand.loc[i], client)
            out(f"\n[{cand.loc[i, 'country_name']} - {display_label(cand.loc[i])}] ({n.source}{'; violations: ' + str(n.violations) if n.violations else ''})\n{n.text}")
    return cand


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--yes", action="store_true", help="confirm the parsed profile without prompting")
    ap.add_argument("--set", action="append", help="edit a field before confirming, e.g. --set industry=software")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--no-narrate", action="store_true")
    a = ap.parse_args(argv)
    if not a.yes and sys.stdin.isatty() and input("Confirm profile? [y/N] ").lower() != "y":
        print("aborted")
        return
    run(a.prompt, a.set, a.top, not a.no_narrate)


if __name__ == "__main__":
    main()

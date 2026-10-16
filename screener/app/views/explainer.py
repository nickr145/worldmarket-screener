"""Label explainer: registered definitions and the out-of-fold tier evidence."""
import json

import pandas as pd
import streamlit as st

from screener.features.run_panel import ROOT, vp
from screener.parse.narrate import LABEL_DEFINITIONS
from screener.score.labels import DISPLAY, TIER_NOTE, label_table


def render():
    st.header("What the labels mean")
    for v in ("expand", "consider", "avoid", "not_rated"):
        st.markdown(f"**{LABEL_DEFINITIONS[v]}**")
    st.dataframe(label_table(), hide_index=True, width="stretch", key="label_map")
    st.caption("The UI shows the display labels. The internal tier names in the table are the ones used in the data files, tests and the registered model card sections. "
               "Within an industry, 'top third' means the highest third of the 25-ish scored cells. Turkey is scored for reference only and is shown as 'Not rated (reference only)' "
               "(its growth is unadjusted for inflation, not CPI-deflated).")
    st.info(TIER_NOTE)
    st.subheader("Did the labels work out of sample?")
    tab = pd.read_csv(vp("reports/verdict_v2_eval.csv"))
    dec = json.load(open(vp("reports/verdict_v2_decision.json")))
    crit = dec["criteria"]
    c = list(crit.values())
    st.markdown(f"- **{DISPLAY['avoid']}** cells realized **{c[0]['est']:+.3f}** higher downside deviation than the rest (95% bootstrap interval {c[0]['lo']:+.3f} to {c[0]['hi']:+.3f}).\n"
                f"- **{DISPLAY['expand']}** cells realized **{c[1]['est'] * 100:+.1f} percentage points** more real revenue growth than the rest (95% bootstrap interval {c[1]['lo'] * 100:+.1f} to {c[1]['hi'] * 100:+.1f}).\n"
                f"- **{DISPLAY['expand']}** cells realized **{c[2]['est']:+.3f}** downside deviation versus the rest (95% bootstrap interval {c[2]['lo']:+.3f} to {c[2]['hi']:+.3f}), i.e. no worse.")
    t = tab.assign(**{"Mean realized": tab.mean_realized.map(lambda x: f"{x:.3f}"), "95% bootstrap interval": tab.apply(lambda r: f"{r.lo:.3f} to {r.hi:.3f}", axis=1)})
    t["tier"] = t.tier.map(DISPLAY).fillna(t.tier)
    t["outcome"] = t.outcome.map({"dd_dev_12m": "12-month downside deviation", "target_h1": "next-year real revenue growth"})
    st.dataframe(t[["sample", "tier", "outcome", "n_cells", "Mean realized", "95% bootstrap interval"]].rename(columns={"sample": "Sample", "tier": "Label", "outcome": "Outcome", "n_cells": "Cell-years"}),
                 hide_index=True, width="stretch", key="tier_table")
    st.warning("Caveats: the growth evidence rests on two as-of years (2023-2024) in one post-pandemic regime; the third criterion is partly mechanical ('Strong momentum, volatility not elevated' excludes the riskiest third by construction); "
               "the rule was revised after seeing a first, stricter rule that labelled every cell with the middle tier (now displayed as Neutral); many specifications were tried on this one dataset.")

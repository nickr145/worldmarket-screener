"""Display labels are a rename in the UI only: the verdict logic and the frozen internal tier names must not change."""
import hashlib

import numpy as np
import pandas as pd

from screener.features.run_panel import MODELS, ROOT, vp
from screener.score.candidates import candidate_table
from screener.score.labels import DISPLAY, NOT_RATED, label_table

KEY = ["country", "industry", "scored_asof_year"]


def test_verdicts_follow_the_registered_v2_rule():
    sc = pd.read_csv(vp("reports/scored_cells.csv"))
    ok = sc[~sc.insufficient_data]
    v2 = np.where(ok.recent_downside_pct >= 2 / 3, "avoid", np.where((ok.recent_growth_pct >= 2 / 3) & ok.recent_downside_pct.notna(), "expand", "consider"))
    v2 = np.where(ok.low_confidence, "consider", v2)  # Turkey: never actionable
    assert (ok.verdict.values == v2).all()
    assert set(sc.verdict) == {"expand", "consider", "avoid", "insufficient_data"}  # internal names still frozen in the CSV


def test_shipped_verdicts_equal_the_frozen_v1_file():
    now = pd.read_csv(vp("reports/scored_cells.csv")).set_index(KEY).verdict
    frozen = pd.read_csv(MODELS / "scored_cells_frozen.csv").set_index(KEY).verdict
    assert now.index.equals(frozen.index) and (now == frozen.loc[now.index]).all()


def test_v1_artifacts_are_untouched():
    """shipped-v1 (reports/scored_cells.csv and models/shipped_v1) is kept as the record of the first run."""
    sha = lambda p: hashlib.sha256(open(ROOT / p, "rb").read()).hexdigest()
    assert sha("reports/scored_cells.csv") == "322535fb7af6dc320a8044ce9d2e4e734fb2361c6e8b47f92086656328541852"
    assert sha("models/shipped_v1/scored_cells_frozen.csv") == "e2c948e4f4bcd6846478b1f7047c8136810d7a582ae32b59a04bed03bdc22bea"


def test_display_labels_are_a_pure_rename():
    assert list(DISPLAY) == ["expand", "consider", "avoid"] and len(set(DISPLAY.values())) == 3
    cand, _ = candidate_table("software", set())
    assert set(cand.verdict) <= set(DISPLAY)  # the table logic still sees internal names
    rows = [l for l in (ROOT / "docs/model_card.md").read_text().splitlines() if l.startswith("| ") and "Display label" not in l and "---" not in l]
    doc = [tuple(c.strip() for c in r.strip("|").split("|"))[:2] for r in rows]
    assert doc == [*[(DISPLAY[k], k) for k in DISPLAY], (NOT_RATED, "consider")] == list(zip(label_table()["Display label"], label_table()["Internal tier name"]))


def test_expand_label_rename_leaves_verdict_assignment_alone():
    assert DISPLAY["expand"] == "Strong momentum, volatility not elevated"
    sc = pd.read_csv(vp("reports/scored_cells.csv"))
    frozen = pd.read_csv(MODELS / "scored_cells_frozen.csv")
    assert sc.verdict.value_counts().to_dict() == frozen.verdict.value_counts().to_dict()
    cand, _ = candidate_table("software", set())  # same cells get the same display label as their internal tier
    assert (cand.verdict.map(DISPLAY).isin(DISPLAY.values())).all()

"""Audit of the built panel (skipped if data/panel/panel.parquet does not exist)."""

import pandas as pd
import pytest

from screener.features.run_panel import vp

P = vp("data/panel/panel.parquet")
pytestmark = pytest.mark.skipif(not P.exists(), reason="panel not built")


def test_no_feature_input_dated_after_decision_date():
    p = pd.read_parquet(P)
    assert (p.features_max_asof_date.dropna() <= p.decision_date[p.features_max_asof_date.notna()]).all()


def test_targets_only_where_cell_ok_and_train_flag_excludes_turkey():
    p = pd.read_parquet(P)
    assert p[(p.status != "ok")].target_h2.isna().all()
    assert not p[p.country == "TR"].train_eligible.any() and p[p.country == "TR"].low_confidence.all()
    assert "TW" not in set(p.country)

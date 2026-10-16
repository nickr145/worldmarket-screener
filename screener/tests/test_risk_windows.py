"""Window-leakage audit for the price-based risk targets."""

import numpy as np
import pandas as pd
import pytest

from screener.features.price_targets import window
from screener.features.run_panel import vp

P = vp("data/panel/x.parquet").parent


def test_synthetic_planted_jumps_stay_on_their_side_of_D():
    idx = pd.bdate_range("2019-01-01", "2022-12-30")
    D = pd.Timestamp("2021-09-30")
    px = {}
    for i in range(6):
        p = pd.Series(100.0 * (1 + 0.0005 * np.arange(len(idx))), index=idx)
        p[p.index > D] *= 1.0
        px[f"S{i}"] = p
    px["S0"] = px["S0"].copy()
    px["S0"][px["S0"].index >= pd.Timestamp("2021-10-04")] *= 10  # planted jump the first business day after D
    px["S1"] = px["S1"].copy()
    px["S1"][px["S1"].index >= pd.Timestamp("2021-09-29")] *= 0.1  # planted crash 1 day BEFORE D
    syms = list(px)
    tr = window(px, syms, D - pd.DateOffset(months=12), D, 12)
    fw = window(px, syms, D, D + pd.DateOffset(months=12), 12)
    assert max(c.index.max() for c in tr.values()) <= D and min(c.index.min() for c in fw.values()) > D
    assert tr["S0"].max() < 0.05  # the +900% jump (after D) is not in the trailing window
    assert fw["S1"].min() > -0.5  # the -90% crash (before D) is not in the forward window


@pytest.mark.skipif(not vp("data/panel/price_targets.parquet").exists(), reason="price targets not built")
@pytest.mark.parametrize("f", ["price_targets", "price_targets_usd"])
def test_real_windows_do_not_overlap_decision_date(f):
    d = pd.read_parquet(vp(f"data/panel/{f}.parquet"))
    t = d.dropna(subset=["trail_last_date"])
    assert (t.trail_last_date <= t.decision_date).all(), "trailing window ends after the decision date"
    for m in (12, 24):
        w = d.dropna(subset=[f"fwd_first_date_{m}m"])
        assert (w[f"fwd_first_date_{m}m"] > w.decision_date).all(), "forward window starts on/before the decision date"
    b = d.dropna(subset=["trail_last_date", "fwd_first_date_12m"])
    assert (b.trail_last_date < b.fwd_first_date_12m).all(), "trailing and forward windows overlap"

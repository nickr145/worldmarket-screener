"""Leakage tests for the panel builder. Synthetic data with planted future values."""
import pandas as pd

from screener.features.build_panel import SNAPSHOT_COLS, TARGETS, build_panel, decision_date

CFG = {"asof_decision": {"month": 9, "day": 30, "year_offset": 1}, "top_n_per_cell": 25, "min_cell_firms": 5,
       "concentration_flag": 0.6, "firm_growth_floor": {"min_prior_revenue_usd": 1e6, "winsorize": [0.0, 1.0]}}
T = 2021
D = decision_date(T, CFG)


def firms(n=6, lag=120):
    rows = []
    for i in range(n):
        for fy in range(2019, 2025):
            fye = pd.Timestamp(f"{fy}-12-31")
            rows.append({"symbol": f"F{i}", "country": "XX", "industry": "ind", "fiscal_year": fy, "fiscal_year_end": fye,
                         "as_of_date": fye + pd.Timedelta(days=lag), "revenue": 100e6 * (1 + 0.1 * (fy - 2019)) * (1 + i / 10),
                         "operating_income": 10e6, "net_income": 5e6, "total_debt": 20e6, "total_assets": 100e6,
                         "rev_usd": 100e6 * (1 + 0.1 * (fy - 2019)) * (1 + i / 10), "infl": 0.02, "deflate": True})
    return pd.DataFrame(rows)


def macro(plant=False):
    rows = [{"iso3": "XXX", "name": n, "year": y, "value": 100.0 + y, "as_of_date": pd.Timestamp(f"{y}-12-31") + pd.Timedelta(days=180)}
            for n in ("gdp_usd", "population") for y in range(2015, 2025)]
    if plant:  # observation dated year T+1 whose release date is after D: must never be a feature at T
        rows += [{"iso3": "XXX", "name": "gdp_usd", "year": T + 1, "value": 9e15, "as_of_date": D + pd.Timedelta(days=30)}]
    return pd.DataFrame(rows)


CELLS = pd.DataFrame([{"country": "XX", "industry": "ind", "iso3": "XXX", "region": "R", "n_listed": 6, "mcap_usd": 1e9,
                       "country_listed": 6, "low_confidence": False, "train_eligible": True}])


def features(panel):
    skip = set(TARGETS) | {c for c in panel if c.startswith("n_target") or c.startswith("target")}
    return panel[[c for c in panel if c not in skip]]


def test_planted_future_values_do_not_enter_features():
    F0 = firms()
    p0, _ = build_panel(F0, macro(), CELLS, CFG, [T])
    F1 = F0.copy()
    # (1) wildly different FY t+1 / t+2 revenue (targets may change, features must not)
    F1.loc[F1.fiscal_year.isin([T + 1, T + 2]), ["revenue", "rev_usd"]] *= 1e4
    p1, _ = build_panel(F1, macro(plant=True), CELLS, CFG, [T])
    pd.testing.assert_frame_equal(features(p0).reset_index(drop=True), features(p1).reset_index(drop=True))
    assert p1.target_h1.iloc[0] != p0.target_h1.iloc[0]  # the plant did reach the target, as it should


def test_reporting_lag_uses_as_of_date_not_fiscal_year_end():
    F = firms()
    late = F.fiscal_year == T
    F.loc[late & (F.symbol.isin(["F0", "F1", "F2", "F3", "F4"])), "as_of_date"] = D + pd.Timedelta(days=1)  # FYE <= D, filing after D
    p, obs = build_panel(F, macro(), CELLS, CFG, [T])
    assert p.n_members.iloc[0] == 1 and p.status.iloc[0] == "insufficient_data"  # only F5 was public at D
    assert set(obs.symbol) == {"F5"}


def test_ranking_ignores_future_revenue():
    F = firms(n=30)
    base, obs0 = build_panel(F, macro(), CELLS, CFG, [T])
    F2 = F.copy()
    F2.loc[(F2.symbol == "F0") & (F2.fiscal_year == T + 2), ["revenue", "rev_usd"]] = 1e15  # F0 would be #1 by future revenue
    _, obs1 = build_panel(F2, macro(), CELLS, CFG, [T])
    assert set(obs0.symbol) == set(obs1.symbol) and len(obs0) == 25


def test_audit_column_never_after_decision_date_and_snapshot_registry():
    p, _ = build_panel(firms(), macro(plant=True), CELLS, CFG, [T])
    assert (p.features_max_asof_date <= p.decision_date).all()
    assert set(SNAPSHOT_COLS) <= set(p.columns)

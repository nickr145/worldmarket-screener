"""Numbers quoted in the README, the model card and the banner must match the evaluation files they come from (shipped-v1.1)."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from screener.app import services as S
from screener.features.run_panel import ROOT, vp

README = (ROOT / "README.md").read_text()
CARD = (ROOT / "docs/model_card.md").read_text()
PLAN = (ROOT / "PLAN.md").read_text()


def row(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v] if not k.endswith("_startswith") else df[df[k[:-11]].str.startswith(v)]
    return df.iloc[0]


def iv(e, lo, hi, d=2):
    return f"{e:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def test_quoted_numbers_match_the_files():
    b, r = pd.read_csv(vp("reports/baselines.csv")), pd.read_csv(vp("reports/risk_targets.csv"))
    b = b[b.spec.str.startswith("PRIMARY") & (b.cv == "country")].set_index("model")
    r = r[r.set.str.startswith("confirmatory") & (r.cv == "country") & (r.target == "dd_dev_12m")].set_index("model")
    g = lambda m, k, f=1: iv(*(b.loc[m, c] * f for c in (k, k + "_lo", k + "_hi")), d=2 if f == 1 else 1)
    d = lambda m, k: iv(*(r.loc[m, c] for c in (k, k + "_lo", k + "_hi")))
    assert f"Persistence Spearman {g('trailing_persistence', 'spearman')}" in README
    assert f"spread {g('trailing_persistence', 'spread', 100)} percentage points" in README
    assert "best growth R2" not in README
    # the R2 sentence must attach each number to the right model (ridge_full is the core of the shipped estimate, not the shipped estimate itself)
    bo = pd.read_csv(vp("reports/boosted.csv"))
    bo = bo[bo.cv == "country"].set_index("model")
    b3 = lambda m: iv(bo.loc[m, "r2"], bo.loc[m, "r2_lo"], bo.loc[m, "r2_hi"])
    assert f"The ridge with trailing growth, the core of the shipped growth estimate (which shrinks it halfway toward the industry mean), has R2 {b3('ridge_full')}. LightGBM's R2 was higher, {b3('lightgbm')}," in README
    assert bo.loc["lightgbm", "r2"] > bo.loc["ridge_full", "r2"] and bo.loc["xgboost", "r2"] < bo.loc["lightgbm", "r2"]
    from screener.score.score_cells import LAMBDA
    assert LAMBDA == 0.5  # "halfway"
    rm = pd.read_csv(vp("reports/promotion_tests.csv")).query("cv == 'country' and metric == 'rmse' and model == 'lightgbm' and vs == 'ridge_full'").iloc[0]
    assert f"in its favour, was +{iv(rm.diff_model_better, rm.lo, rm.hi, 3)}, which includes zero" in README
    assert rm.diff_model_better > 0 and rm.lo < 0 < rm.hi
    assert f"R2 {g('ridge_macro', 'r2')} against {g('industry_mean', 'r2')} for the industry mean" in README
    assert f"about {b.spearman.trailing_persistence:.2f}" in CARD
    assert f"R2 about {r.r2.trailing_persistence:.2f}, Spearman about {r.spearman.trailing_persistence:.2f}" in CARD
    assert (S.SKILL_GROWTH, S.SKILL_DOWNSIDE) == (f"{b.spearman.trailing_persistence:.2f}", f"{r.spearman.trailing_persistence:.2f}")
    assert f"R2 {d('trailing_persistence', 'r2')}, Spearman {d('trailing_persistence', 'spearman')}" in README
    assert (b.n_rows.iloc[0], b.n_countries.iloc[0]) == (181, 29) and "29 of the 30 countries are in the evaluation sample" in README
    # boosted models: growth Spearman vs persistence, and downside Spearman vs persistence
    pt = pd.read_csv(vp("reports/promotion_tests.csv"))
    pt = pt[(pt.cv == "country") & (pt.metric == "spearman")].set_index("model")
    rp = pd.read_csv(vp("reports/risk_promotion_tests.csv"))
    rp = rp[rp.set.str.startswith("confirmatory") & (rp.cv == "country") & (rp.target == "dd_dev_12m") & (rp.metric == "spearman")].set_index("model")
    t = lambda x, m: iv(x.loc[m, "diff_model_better"], x.loc[m, "lo"], x.loc[m, "hi"])
    assert f"XGBoost {t(pt, 'xgboost')}, LightGBM {t(pt, 'lightgbm')}" in README
    assert f"{t(rp, 'xgboost')} and {t(rp, 'lightgbm')}" in README
    assert (rp.loc[["xgboost", "lightgbm"], "hi"] < 0).all()  # "significantly worse" on downside rank
    assert not pt.favours_model_significantly.any()  # "not promoted"
    # controls
    rc = pd.read_csv(vp("reports/risk_controls.csv"))
    rc = rc[rc.target == "dd_dev_12m"]
    comp = rc[rc.sensitivity.str.startswith("composition") & (rc.kind == "paired persistence-minus-composition_ridge") & (rc.model == "spearman")].iloc[0]
    usd = rc[rc.sensitivity.str.startswith("USD") & (rc.kind == "metrics") & (rc.model == "trailing_persistence")].iloc[0]
    assert "cell-composition control (firm count and top-firm share;" in README and "firm-count control" not in README
    from screener.model.risk_controls import COMP
    assert COMP == ["n_members", "top1_share"]
    assert f"Spearman advantage +{iv(comp.diff_persistence_better, comp.diff_lo, comp.diff_hi)}" in README
    assert f"(R2 {iv(usd.r2, usd.r2_lo, usd.r2_hi)})" in README
    # registered v2 criterion
    dec = json.load(open(vp("reports/verdict_v2_decision.json")))
    assert dec["passed"]
    c = list(dec["criteria"].values())
    assert f"+{c[0]['est']:.3f} [+{c[0]['lo']:.3f}, +{c[0]['hi']:.3f}] more downside deviation" in README
    assert f"+{c[1]['est'] * 100:.1f} [+{c[1]['lo'] * 100:.1f}, +{c[1]['hi'] * 100:.1f}] percentage points more real growth" in README
    assert f"{c[2]['est']:.3f} [{c[2]['lo']:.3f}, {c[2]['hi']:.3f}] downside" in README
    ev = pd.read_csv(vp("reports/verdict_v2_eval.csv"))
    ev = ev[ev["sample"].str.startswith("J") & (ev.outcome == "target_h1")].set_index("tier")
    assert set(ev["sample"]) == {"J (2023-2024, joint)"}
    sc = pd.read_csv(vp("reports/scored_cells.csv"))
    assert set(sc.verdict_v1[sc.verdict_v1 != "insufficient_data"]) == {"consider"}  # "v1 labelled every cell consider"
    assert sc.country.nunique() == 30 and sc.industry.nunique() == 4 and "4 industries and 30 countries" in README
    cov = sc.groupby("industry").industry_coverage_loco.first()
    assert f"coverage {cov.min():.2f}-{cov.max():.2f}" in CARD
    # snapshot and version
    snap = pd.Timestamp(json.load(open(vp("reports/data_snapshot.json")))["snapshot_date"])
    assert f"data snapshot {snap.day} {snap:%B %Y}" in README and "`shipped-v1.1`" in README
    # v1 -> v1.1 comparison quoted in the Process paragraph
    old = pd.read_csv(ROOT / "reports/scored_cells.csv")
    assert (old.verdict == sc.verdict).all()
    assert f"largest growth-estimate change {(old.model_estimate - sc.model_estimate).abs().max() * 100:.3f} percentage points" in README
    assert f"{(old.drop(columns='country').fillna(-9).ne(sc[old.columns].drop(columns='country').fillna(-9)).any(axis=1)).sum()} of {len(sc)} cells changed" in README
    # HCLTECH.NS: the false concentration flag on India software cleared
    oi, ni = (x[(x.country == "IN") & (x.industry == "software")].iloc[0] for x in (old, sc))
    thr = yaml.safe_load(open(ROOT / "screener/config/settings.yaml"))["concentration_flag"]
    assert thr == 0.6 and oi.top1_share > thr >= ni.top1_share and oi.concentrated and not ni.concentrated
    assert f"top-firm share {oi.top1_share * 100:.1f}% to about {ni.top1_share * 100:.0f}%" in README and "over 60% of revenue" in README
    # limitations: barely-excluded-zero intervals and non-bank financials
    usd = pd.read_csv(vp("reports/risk_controls.csv")).query("sensitivity.str.startswith('USD') and kind == 'metrics' and model == 'trailing_persistence' and target == 'dd_dev_12m'", engine="python").iloc[0]
    rp = pd.read_csv(vp("reports/risk_promotion_tests.csv"))
    lp = rp[rp.set.str.startswith("confirmatory") & (rp.cv == "country") & (rp.target == "dd_dev_12m") & (rp.metric == "spearman") & (rp.model == "lightgbm")].iloc[0]
    assert 0 < usd.r2_lo < 0.02 and -0.02 < lp.hi < 0
    assert f"USD-price downside R2 lower bound is {usd.r2_lo:.2f} and LightGBM's downside Spearman upper bound is {lp.hi:.2f}" in README
    one = sc.groupby("industry").agg(hw=("model_estimate_halfwidth", "max"), n=("insufficient_data", lambda s: int((~s.astype(bool)).sum())))
    assert one.hw.idxmax() == "non_bank_financials" and one.n.idxmin() == "non_bank_financials"
    tr = sc[sc.country == "TR"]
    assert tr.low_confidence.all() and (tr.verdict == "consider").all()
    # Korea healthcare sampling (scope limits in README and PLAN, model card)
    g2 = pd.read_csv(vp("reports/gate2_cells.csv")).query("country == 'KR' and industry == 'healthcare'").iloc[0]
    kr = int(g2.firms_4fy_revenue_and_cpi)
    members = pd.read_parquet(vp("data/panel/panel.parquet")).query("country == 'KR' and industry == 'healthcare' and as_of_year >= 2022").n_members
    assert g2.firms_fetched == 60 and (members == 25).all()
    for txt in (README, PLAN):
        assert f"{kr} of the 60 fetched firms have four fiscal years of revenue and CPI, and the cell has 25 members in every as-of year" in txt
        assert "only 32 of the 60" not in txt
    assert f"{kr} of 60 fetched firms have four fiscal years of revenue and CPI" in CARD and f"the final cache has {kr} of 60" not in README and f"which has {kr} of 60" in README


def test_readme_layout_and_visible_numbers():
    head, rest = README.split("<details>", 1)
    visible = head[head.index("## What I found"):]
    assert len(visible.split()) < 200  # the part a reader sees without expanding
    assert rest.startswith("\n<summary>Evidence and caveats</summary>\n\n") and rest.count("</details>") == 1
    assert README.index("</details>") < README.index("### Labels")
    b = pd.read_csv(vp("reports/baselines.csv"))
    b = b[b.spec.str.startswith("PRIMARY") & (b.cv == "country")].set_index("model")
    bo = pd.read_csv(vp("reports/boosted.csv"))
    bo = bo[bo.cv == "country"].set_index("model")
    r = pd.read_csv(vp("reports/risk_targets.csv"))
    r = r[r.set.str.startswith("confirmatory") & (r.cv == "country") & (r.target == "dd_dev_12m")].set_index("model")
    rp = pd.read_csv(vp("reports/risk_promotion_tests.csv"))
    rp = rp[rp.set.str.startswith("confirmatory") & (rp.cv == "country") & (rp.target == "dd_dev_12m") & (rp.metric == "spearman")].set_index("model")
    n_crit = len(json.load(open(vp("reports/verdict_v2_decision.json")))["criteria"])
    for line in (
        f"Recent growth ranks cells (Spearman {b.spearman.trailing_persistence:.2f}) but explains little in levels (ridge R2 {bo.r2.ridge_full:.2f}).",
        f"(R2 {b.r2.ridge_macro:.2f} against {b.r2.industry_mean:.2f} for the industry mean)",
        f"(Spearman difference {rp.diff_model_better.xgboost:.2f} and {rp.diff_model_better.lightgbm:.2f})",
        f"(R2 {r.r2.trailing_persistence:.2f}, Spearman {r.spearman.trailing_persistence:.2f})",
        f"(all {n_crit} criteria)",
        "4 industries and 30 countries",
    ):
        assert line in visible, line
    assert "[" not in visible.replace("[`", "")  # no intervals in the visible lines


def test_verdict_loose_column_note():
    """verdict_loose_no_interval_rule is 'consider' everywhere; the model card says what it was meant to hold."""
    for f in ("reports/scored_cells.csv", "reports/scored_cells_v1_1.csv"):
        d = pd.read_csv(ROOT / f)
        assert (d.verdict_loose_no_interval_rule == "consider").all()
        keep = d.verdict_v1 != "insufficient_data"
        assert (d.verdict_loose_no_interval_rule[keep] == d.verdict_v1[keep]).all()  # a copy of verdict_v1 before the insufficient-data overwrite
        elig = (~d.insufficient_data.astype(bool)) & (d.country != "TR")
        med = d[elig].groupby("industry")[["recent_growth_signal", "recent_downside_signal"]].median()
        m = d.join(med, on="industry", rsuffix="_med")
        up = (m.recent_growth_signal > m.recent_growth_signal_med) & (m.recent_downside_signal < m.recent_downside_signal_med)
        dn = (m.recent_growth_signal < m.recent_growth_signal_med) & (m.recent_downside_signal > m.recent_downside_signal_med)
        v = pd.Series(np.where(up, "expand", np.where(dn, "avoid", "consider")), index=d.index).where(~d.insufficient_data.astype(bool), "insufficient_data")
        c = v.value_counts()
        assert f"{c['expand']} expand / {c['consider']} consider / {c['avoid']} avoid among the {len(d)} cells, {c['insufficient_data']} insufficient data" in CARD
        assert f"{c['expand']} expand / {c['consider']} consider / {c['avoid']} avoid" in PLAN  # the count printed when v2 was registered


def test_week2_week3_counts_against_final_files():
    cc = pd.read_csv(ROOT / "reports/cell_counts.csv")
    th = cc[cc.country == "TH"].set_index("industry").domestic
    assert th.tolist() == [41, 41, 28, 28] and f"Thai domestic counts {'/'.join(map(str, th))}" in PLAN
    assert (cc.domestic >= 5).sum() == 104 and len(cc) == 128
    g = pd.read_csv(vp("reports/gate2_cells.csv"))
    u = g[g.usable]
    assert f"{len(u)} of {len(g)} cells usable" in PLAN
    pan = pd.read_parquet(vp("data/panel/panel.parquet"))
    h2 = pan[pan.target_h2.notna()]
    h2t = h2[h2.trailing_growth.notna()]
    assert h2.groupby(["country", "industry"]).ngroups == 96 and h2t.groupby(["country", "industry"]).ngroups == 92 and set(h2t.as_of_year) == {2023}
    assert "96 cells in the final panel (92 with trailing growth, 2023 only)" in PLAN
    # countries
    cfg = yaml.safe_load(open(ROOT / "screener/config/countries.yaml"))["countries"]
    assert sorted(set(cfg) - set(pan.country)) == ["AR", "TW"] and pan.country.nunique() == 30
    assert not {"AE", "CO", "EG", "PH", "VN", "IE", "NG", "KE"} & set(cfg)
    assert cc[cc.country == "AR"].domestic.max() < 5
    assert "eight further candidates (AE, CO, EG, PH, VN, IE, NG, KE)" in CARD and "two configured countries are not in the panel" in CARD
    assert "Argentina is configured but has fewer than 5 domestic listed firms" in PLAN and "Argentina is configured but has fewer than 5 domestic listed firms" in README
    # old-vs-new overlap (12 of 13 case countries, all of as-of 2022-2024)
    old13 = ["AR", "BR", "CA", "CN", "DE", "GB", "ID", "IN", "JP", "KR", "PL", "SG", "US"]
    ok = pan[(pan.industry == "retail") & pan.country.isin(old13) & pan.as_of_year.between(2022, 2024) & (pan.status == "ok")].groupby("country").size()
    assert (ok == 3).sum() == 12 and "AR" not in ok.index
    assert g[(g.country == "AU") & (g.industry == "retail")].firms_4fy_revenue_and_cpi.iloc[0] == 4


@pytest.mark.skipif(not (ROOT / "reports/firm_universe.csv").exists() or not (ROOT / "data/raw/domicile.parquet").exists(), reason="firm-level files are not in the public export")
def test_korea_domicile_counts():
    dom = pd.read_parquet(ROOT / "data/raw/domicile.parquet")
    assert dom[(dom.domicile == "ERR") & dom.symbol.str.endswith(".KQ")].shape[0] == 94
    fu = pd.read_csv(ROOT / "reports/firm_universe.csv")
    kr = fu[fu.country == "KR"]
    assert kr.domicile_assumed.sum() == 79 and set(kr[kr.domicile_assumed].exchange) == {"KOE"}
    n = sum(g[g.domestic | g.domicile_assumed].sort_values("mcap", ascending=False).head(25).domicile_assumed.sum() for _, g in kr.groupby("industry"))
    assert n == 13
    assert "94 `.KQ` lookups are empty in the raw domicile cache, 79 remain after name de-duplication, and 13 of them rank" in PLAN


def expand_sample():
    """The v2 growth test's joint sample (j) and its expand tier (e), built exactly as verdict_v2_eval builds them."""
    from screener.score.verdict_v2_eval import tiers
    p = pd.read_parquet(vp("data/panel/panel.parquet"))[["country", "industry", "as_of_year", "trailing_growth", "target_h1"]]
    pt = pd.read_parquet(vp("data/panel/price_targets.parquet"))  # dataset() only adds macro columns, which the tiers do not use
    pt = pt[~pt.country.isin(yaml.safe_load(open(ROOT / "screener/config/settings.yaml"))["exclude_from_training"])]
    d = pt.merge(p, on=["country", "industry", "as_of_year"], how="left")
    d = tiers(d[~d.membership_lookahead & d.as_of_year.between(2022, 2024) & d.trail_dd_dev_12m.notna()])
    j = d[d.dd_dev_12m.notna() & d.target_h1.notna() & d.trailing_growth.notna() & d.as_of_year.between(2023, 2024)]
    return j[j.tier == "expand"], j


def test_expand_tier_cell_years_and_countries():
    e, j = expand_sample()
    ev = pd.read_csv(vp("reports/verdict_v2_eval.csv"))
    ev = ev[ev["sample"].str.startswith("J") & (ev.outcome == "target_h1")].set_index("tier")
    assert (len(e), len(j)) == (int(ev.loc["expand", "n_cells"]), int(ev.n_cells.sum()))
    assert e.as_of_year.nunique() == 2
    assert f"{len(e)} cell-years from {e.country.nunique()} countries (the {len(j)} cell-years they are compared with span {j.country.nunique()})" in CARD
    assert "28 countries" not in CARD.split("4. **`expand` growth evidence")[1].split("\n")[0].split("(the")[0]
    assert f"The growth evidence rests on {len(e)} cell-years from {e.country.nunique()} countries over two evaluation years (2023 and 2024)." in README
    assert "two evaluation years (170" not in README


def test_dropped_countries_wording():
    """Each never-covered or dropped country is named once in the README's scope limits, with the reason the config records."""
    import re
    cfg = yaml.safe_load(open(ROOT / "screener/config/countries.yaml"))
    dropped = cfg["dropped_countries"]
    assert set(dropped) == {"AE", "CO", "EG", "PH", "VN", "IE", "NG", "KE"}
    assert "zero firms" in dropped["AE"] and "zero firms" in dropped["IE"] and all(dropped[k] == "same as AE" for k in ("CO", "EG", "PH", "VN")) and all("no yfinance screener region" in dropped[k] for k in ("NG", "KE"))
    assert cfg["countries"]["TW"] and "TW" in yaml.safe_load(open(ROOT / "screener/config/settings.yaml"))["drop_from_panel"]
    para = README[README.index("(and the dropped countries:"):].split(") are absent")[0]
    for name in ("AE", "CO", "EG", "PH", "VN", "IE", "Nigeria", "Kenya", "Taiwan", "Argentina"):
        assert len(re.findall(rf"\b{name}\b", README.split("\n## The original ")[0])) == 1, name  # the frozen original-case section (cut from the export) also mentions Argentina
    assert "AE, CO, EG, PH, VN have zero screener coverage; IE has none either; Nigeria and Kenya have no yfinance region; Taiwan lacks structural macro data; Argentina is configured but has fewer than 5 domestic listed firms in every industry" in para

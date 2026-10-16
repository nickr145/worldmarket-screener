"""Live scoring with the SHIPPED models (PLAN.md 'Shipped-model specification'). No boosted models, no LLM.

 growth : ranking/verdict signal = trailing-growth persistence; point = industry_mean + 0.5 * (ridge_full - industry_mean);
          80% split-conformal interval on leave-country-out residuals.
 downside deviation : rank = persistence of the trailing 12m measure; point = persistence; 80% conformal interval.
 max drawdown : ordering only (no numeric estimate).
 verdict : per-industry medians; interval straddling the threshold -> consider.
python -m screener.score.score_cells -> reports/scored_cells.csv
"""
import json
import math

import numpy as np
import pandas as pd

from screener.features.run_panel import ROOT, vp, load_cfg
from screener.model.baselines import SHARED, SPECIFIC, load, ridge_fit_predict
from screener.model.risk_models import dataset

LAMBDA, LEVEL = 0.5, 0.8
PRIMARY = "PRIMARY h1 real | trailing | no-TR | no-biotech"


def shipped_point(tr, te):
    """tr/te frames with log_gdp_pc etc.; returns (industry_mean, ridge_full, shrunk) arrays for te."""
    im = te.industry.map(tr.groupby("industry").y.mean()).fillna(tr.y.mean()).values
    rf = im.copy()
    for ind in te.industry.unique():
        m = (te.industry == ind).values
        r = ridge_fit_predict(tr[tr.industry == ind], te[m], SPECIFIC[ind] + SHARED + ["trailing_growth"])
        if r is not None:
            rf[m] = r
    return im, rf, im + LAMBDA * (rf - im)


def conformal_q(res_other):
    n = len(res_other)
    k = min(1.0, math.ceil((n + 1) * LEVEL) / n)
    return float(np.quantile(np.abs(res_other), k))


def loco(p, pred_fn):
    """Leave-one-country-out residuals y - pred(country held out)."""
    res = np.full(len(p), np.nan)
    for c in p.country.unique():
        te = (p.country == c).values
        res[te] = p.y.values[te] - pred_fn(p[~te], p[te])
    return res


def coverage(p, res):
    """Interval for country c uses q from the OTHER countries' residuals; coverage measured on c (approximate: those residuals come from
    models that saw c in training; a fully nested version would refit per pair)."""
    hit = np.zeros(len(p), bool)
    for c in p.country.unique():
        te = (p.country == c).values
        hit[te] = np.abs(res[te]) <= conformal_q(res[~te])
    rng = np.random.default_rng(0)
    cs = p.country.unique()
    idx = {c: np.where(p.country.values == c)[0] for c in cs}
    boots = [hit[np.concatenate([idx[c] for c in rng.choice(cs, len(cs))])].mean() for _ in range(1000)]
    return hit.mean(), np.quantile(boots, 0.025), np.quantile(boots, 0.975), hit


def main():
    st, countries, _ccys = load_cfg()
    panel = pd.read_parquet(vp("data/panel/panel.parquet"))
    panel = panel.assign(log_gdp_pc=np.log(panel.gdp_pc_usd))
    # ---- growth: fit on all labelled rows
    lab, _ = load(PRIMARY)
    res_g = loco(lab, lambda tr, te: shipped_point(tr, te)[2])
    qg = conformal_q(res_g)  # pooled (v1, kept for reference)
    qg_ind = {i: conformal_q(res_g[(lab.industry == i).values]) for i in lab.industry.unique()}  # v2: industry-specific
    cov_ind = {}
    for i in lab.industry.unique():
        m = (lab.industry == i).values
        cov_ind[i] = coverage(lab[m].reset_index(drop=True), res_g[m])
    hit_all = np.zeros(len(lab), bool)
    for i in cov_ind:
        hit_all[(lab.industry == i).values] = cov_ind[i][3]
    cov_g = (hit_all.mean(),) + cov_ind[next(iter(cov_ind))][1:3] + (hit_all,)
    # ---- downside deviation: persistence, labelled = confirmatory rows
    d = dataset()
    dl = d[~d.membership_lookahead & d.as_of_year.between(2022, 2024) & d.dd_dev_12m.notna() & d.trail_dd_dev_12m.notna()].reset_index(drop=True)
    dl["y"] = dl.dd_dev_12m
    res_d = dl.y.values - dl.trail_dd_dev_12m.values
    qd = conformal_q(res_d)
    cov_d = coverage(dl, res_d)
    # ---- rows to score: latest as-of year per cell with a usable growth signal
    cand = panel[(panel.status == "ok") & panel.trailing_growth.notna()].sort_values("as_of_year")
    latest = cand.groupby(["country", "industry"]).tail(1).copy()
    allcells = panel.groupby(["country", "industry"]).agg(region=("region", "first"), low_confidence=("low_confidence", "first")).reset_index()
    out = allcells.merge(latest.drop(columns=["region", "low_confidence"]), on=["country", "industry"], how="left")
    out["scored_asof_year"] = out.as_of_year
    top = st["as_of_years"][-1]
    out["stale_financials"] = out.scored_asof_year.notna() & (out.scored_asof_year < top)
    sc = out[out.trailing_growth.notna()].copy()
    sc["y"] = np.nan
    im, rf, pt = shipped_point(lab, sc)
    out.loc[sc.index, "growth_point"] = pt
    out["growth_signal"] = out.trailing_growth
    out["growth_q"] = out.industry.map(qg_ind)
    out["growth_lo"], out["growth_hi"] = out.growth_point - out.growth_q, out.growth_point + out.growth_q
    out["industry_coverage_loco"] = out.industry.map({i: c[0] for i, c in cov_ind.items()})
    out["interval_reliable"] = out.industry_coverage_loco.between(0.70, 0.90)
    # risk signal from price targets at the scored as-of year
    pt_ = pd.read_parquet(vp("data/panel/price_targets.parquet"))[["country", "industry", "as_of_year", "trail_dd_dev_12m", "trail_mdd_12m"]]
    out = out.merge(pt_.rename(columns={"as_of_year": "scored_asof_year"}), on=["country", "industry", "scored_asof_year"], how="left")
    out["dd_signal"] = out.trail_dd_dev_12m
    out["dd_point"] = out.dd_signal
    out["dd_lo"], out["dd_hi"] = (out.dd_point - qd).clip(lower=0), out.dd_point + qd
    # flags
    fo = pd.read_parquet(vp("data/panel/firm_obs.parquet"))[["symbol", "country", "industry", "t"]]
    fu = pd.read_csv(ROOT / "reports/firm_universe.csv").set_index("symbol")
    fo["assumed"] = fo.symbol.map(fu.domicile_assumed)
    share = fo.groupby(["country", "industry", "t"]).assumed.mean().rename("domicile_assumed_share").reset_index().rename(columns={"t": "scored_asof_year"})
    out = out.merge(share, on=["country", "industry", "scored_asof_year"], how="left")
    g2 = pd.read_csv(vp("reports/gate2_cells.csv"))[["country", "industry", "usable"]].rename(columns={"usable": "gate2_usable"})
    out = out.merge(g2, on=["country", "industry"], how="left")
    out["gate2_usable"] = out.gate2_usable.fillna(False)
    out["insufficient_data"] = out.growth_signal.isna()
    out["concentrated"] = out.top1_share > st["concentration_flag"]
    out["train_eligible"] = ~out.country.isin(st["exclude_from_training"])
    # ---- ranks and thresholds within industry (eligible = scored, not Turkey)
    elig = (~out.insufficient_data) & out.train_eligible
    for col, name, asc in (("growth_signal", "growth_rank", False), ("dd_signal", "dd_rank", True), ("trail_mdd_12m", "mdd_order_rank", False)):
        out[name] = out[elig].groupby("industry")[col].rank(ascending=asc, method="min")
    out["n_ranked_in_industry"] = out.groupby("industry").growth_rank.transform("max")
    med = out[elig].groupby("industry").agg(growth_thr=("growth_point", "median"), dd_thr=("dd_point", "median"),
                                           g_sig_med=("growth_signal", "median"), d_sig_med=("dd_signal", "median"))
    out = out.merge(med, on="industry", how="left")
    g_up, g_dn = out.growth_signal > out.g_sig_med, out.growth_signal < out.g_sig_med  # direction from the ranking signal (persistence)
    d_good, d_bad = out.dd_signal < out.d_sig_med, out.dd_signal > out.d_sig_med
    g_clear_up, g_clear_dn = out.growth_lo > out.growth_thr, out.growth_hi < out.growth_thr  # interval excludes the threshold
    d_clear_good, d_clear_bad = out.dd_hi < out.dd_thr, out.dd_lo > out.dd_thr
    loose_exp, loose_av = g_up & d_good, g_dn & d_bad
    strict_exp, strict_av = loose_exp & g_clear_up & d_clear_good, loose_av & g_clear_dn & d_clear_bad
    out["verdict_v1"] = np.where(strict_exp, "expand", np.where(strict_av, "avoid", "consider"))
    out["no_risk_signal"] = out.dd_signal.isna() & ~out.insufficient_data
    # ---- v2 (registered in PLAN.md): persistence tiers within industry among eligible cells
    out["pct_growth"] = out[elig].groupby("industry").growth_signal.transform(lambda x: (x.rank() - 1) / (x.notna().sum() - 1))
    out["pct_dd"] = out[elig].groupby("industry").dd_signal.transform(lambda x: (x.rank() - 1) / (x.notna().sum() - 1))
    v2 = np.where(out.pct_dd >= 2 / 3, "avoid", np.where((out.pct_growth >= 2 / 3) & out.pct_dd.notna(), "expand", "consider"))
    passed = json.load(open(vp("reports/verdict_v2_decision.json")))["passed"]
    out["verdict"] = v2 if passed else "unranked"
    out.loc[~out.train_eligible & ~out.insufficient_data, "verdict"] = "consider" if passed else "unranked"  # Turkey: never actionable
    out.loc[out.insufficient_data, "verdict"] = "insufficient_data"
    out["verdict_loose_no_interval_rule"] = out["verdict_v1"]
    out.loc[out.insufficient_data, "verdict_v1"] = "insufficient_data"
    # ---- freshness: share of the cell's firms with a known FY2024 that also have FY2025 known by the 2025 decision date
    from screener.features.build_panel import decision_date
    from screener.features.run_panel import load_macro, prepare
    F, _ = prepare(st, countries, _ccys, load_macro())
    D = decision_date(st["as_of_years"][-1], st)
    ok_ = F[(F.revenue > 0) & (F.as_of_date <= D)]
    s24 = ok_[ok_.fiscal_year == st["as_of_years"][-1] - 1].groupby(["country", "industry"]).symbol.agg(set)
    s25 = ok_[ok_.fiscal_year == st["as_of_years"][-1]].groupby(["country", "industry"]).symbol.agg(set)
    fresh = {k: len(v & s25.get(k, set())) / len(v) for k, v in s24.items()}
    out["data_freshness"] = [fresh.get((c, i), np.nan) for c, i in zip(out.country, out.industry)]
    out["stale_firms_flag"] = out.data_freshness < 0.60
    out["low_confidence_flag"] = out.low_confidence | out.concentrated | (out.domicile_assumed_share.fillna(0) > 0)
    cols = ["country", "industry", "region", "scored_asof_year", "stale_financials", "gate2_usable", "insufficient_data", "n_members",
            "growth_signal", "growth_rank", "n_ranked_in_industry", "pct_growth", "growth_point", "growth_lo", "growth_hi", "growth_q", "interval_reliable", "industry_coverage_loco", "growth_thr",
            "dd_signal", "dd_rank", "pct_dd", "dd_point", "dd_lo", "dd_hi", "dd_thr", "mdd_order_rank", "verdict", "verdict_v1", "verdict_loose_no_interval_rule",
            "low_confidence", "concentrated", "top1_share", "domicile_assumed_share", "data_freshness", "stale_firms_flag", "no_risk_signal", "low_confidence_flag"]
    from screener.score.labels import RENAME, add_outside
    out = add_outside(out)
    out = out[cols + ["signal_outside_interval", "signal_outside_direction"]].sort_values(["industry", "country"]).reset_index(drop=True)
    out = out.rename(columns=RENAME)  # recent_*_signal (rank) vs model_estimate (interval) are separate labelled fields everywhere
    out.to_csv(vp("reports/scored_cells.csv"), index=False)
    return out, dict(qg=qg, qg_ind=qg_ind, cov_ind=cov_ind, qd=qd, cov_g=cov_g, cov_d=cov_d, lab=lab, dl=dl, res_g=res_g)


if __name__ == "__main__":
    out, info = main()
    pd.set_option("display.width", 250)
    print("scored cells:", int((~out.insufficient_data).sum()), "of", len(out), "| stale_financials:", int(out.stale_financials.sum()))
    print("\nv2 verdict counts by industry:"); print(pd.crosstab(out.industry, out.verdict, margins=True).to_string())
    print("\nv1 (registered strict rule) for reference:"); print(pd.crosstab(out.industry, out.verdict_v1, margins=True).to_string())
    lab = info["lab"]
    print("\nGrowth conformal coverage per industry (leave-country-out, nominal 0.80; industry-specific half-width):")
    cv = pd.DataFrame({i: {"half_width": info["qg_ind"][i], "coverage": c[0], "ci_lo": c[1], "ci_hi": c[2], "n_rows": int((lab.industry == i).sum()),
                           "interval_reliable": bool(0.70 <= c[0] <= 0.90)} for i, c in info["cov_ind"].items()}).T
    print(cv.round(3).to_string())
    print(f"pooled coverage with industry-specific widths: {info['cov_g'][0]:.3f}; downside deviation (pooled half-width {info['qd']:.3f}): {info['cov_d'][0]:.3f} [{info['cov_d'][1]:.3f}, {info['cov_d'][2]:.3f}]")
    print("\nFreshness (share of firms with FY2024 that also have FY2025 by the 2025 decision date):")
    f = out[~out.insufficient_data].data_freshness
    print(f"  cells scored {len(f)} | mean {f.mean():.3f} | median {f.median():.3f} | min {f.min():.3f} | share of cells <60%: {(f < 0.6).mean():.3f} ({int((f < 0.6).sum())} cells)")
    print(out[out.stale_firms_flag][["country", "industry", "data_freshness", "n_members"]].to_string(index=False))
    print("flags: low_confidence", int(out.low_confidence.sum()), "| concentrated", int(out.concentrated.sum()), "| domicile_assumed_share>0", int((out.domicile_assumed_share > 0).sum()),
          "| interval unreliable cells", int((~out.interval_reliable & ~out.insufficient_data).sum()), "| no_risk_signal", int(out.no_risk_signal.sum()))

"""Out-of-fold evaluation of verdict rule v2, exactly as registered in PLAN.md (no threshold changes after seeing results).
python -m screener.score.verdict_v2_eval -> reports/verdict_v2_eval.csv, reports/verdict_v2_decision.json"""
import json

import numpy as np
import pandas as pd

from screener.features.run_panel import ROOT, vp
from screener.model.risk_models import dataset

B, SEED = 1000, 0


def pct(s):
    n = s.notna().sum()
    return (s.rank(method="average") - 1) / (n - 1) if n > 1 else s * 0


def tiers(d):
    """Add v2 tiers from the persistence signals; ranks within industry x as-of year among cells with the signal."""
    d = d.copy()
    g = d.groupby(["industry", "as_of_year"])
    d["pct_dd"] = g.trail_dd_dev_12m.transform(pct)
    d["pct_g"] = g.trailing_growth.transform(pct)
    top_dd, top_g = d.pct_dd >= 2 / 3, d.pct_g >= 2 / 3
    d["tier"] = np.where(top_dd, "avoid", np.where(top_g, "expand", "consider"))
    d.loc[d.pct_dd.isna(), "tier"] = np.nan
    return d


def demean(d, cols):
    for c in cols:
        d[c + "_dm"] = d[c] - d.groupby(["industry", "as_of_year"])[c].transform("mean")
    return d


def boot_diff(d, mask_a, col, draws, idx):
    """Mean(col | mask) - mean(col | ~mask) for each country-resampled draw; returns (estimate, lo, hi)."""
    m = mask_a.values
    v = d[col].values

    def f(sel):
        a, b = v[sel][m[sel]], v[sel][~m[sel]]
        return a.mean() - b.mean() if len(a) and len(b) else np.nan
    est = f(np.arange(len(d)))
    bs = np.array([f(x) for x in draws])
    return est, np.nanquantile(bs, 0.025), np.nanquantile(bs, 0.975)


def main():
    d = dataset()
    p = pd.read_parquet(vp("data/panel/panel.parquet"))[["country", "industry", "as_of_year", "trailing_growth", "target_h1", "status", "concentrated"]]
    d = d.merge(p, on=["country", "industry", "as_of_year"], how="left")
    d = d[~d.membership_lookahead & d.as_of_year.between(2022, 2024) & d.trail_dd_dev_12m.notna()].copy()
    d = tiers(d)
    rng = np.random.default_rng(SEED)
    # sample A: downside only, as-of 2022-2024; sample Bj: joint, as-of 2023-2024
    A = demean(d[d.dd_dev_12m.notna()].reset_index(drop=True), ["dd_dev_12m"])
    J = d[d.dd_dev_12m.notna() & d.target_h1.notna() & d.trailing_growth.notna() & d.as_of_year.between(2023, 2024)].reset_index(drop=True)
    J = demean(J, ["dd_dev_12m", "target_h1"])
    res, tab = {}, []
    for name, S in (("A (2022-2024, downside)", A), ("J (2023-2024, joint)", J)):
        cs = S.country.unique()
        idx = {c: np.where(S.country.values == c)[0] for c in cs}
        draws = [np.concatenate([idx[c] for c in rng.choice(cs, len(cs))]) for _ in range(B)]
        res[name] = (S, draws)
        for t in ("avoid", "expand", "consider"):
            m = S.tier == t
            for col in ["dd_dev_12m"] + (["target_h1"] if "J" in name else []):
                v = S[col].values
                bs = [v[x][m.values[x]].mean() for x in draws if m.values[x].any()]
                tab.append({"sample": name, "tier": t, "outcome": col, "n_cells": int(m.sum()), "mean_realized": v[m.values].mean(),
                            "lo": np.quantile(bs, 0.025), "hi": np.quantile(bs, 0.975)})
    tab = pd.DataFrame(tab)
    A_, dA = res["A (2022-2024, downside)"]
    J_, dJ = res["J (2023-2024, joint)"]
    c1 = boot_diff(A_, A_.tier == "avoid", "dd_dev_12m_dm", dA, None)       # avoid minus rest, realized downside (demeaned)
    c2 = boot_diff(J_, J_.tier == "expand", "target_h1_dm", dJ, None)        # expand minus rest, realized growth (demeaned)
    c3 = boot_diff(J_, J_.tier == "expand", "dd_dev_12m_dm", dJ, None)       # expand minus rest, realized downside (demeaned)
    raw = {"avoid_dd_raw": boot_diff(A_, A_.tier == "avoid", "dd_dev_12m", dA, None), "expand_growth_raw": boot_diff(J_, J_.tier == "expand", "target_h1", dJ, None),
           "expand_dd_raw": boot_diff(J_, J_.tier == "expand", "dd_dev_12m", dJ, None)}
    crit = {"1 avoid: realized downside minus rest, CI > 0": (c1, c1[1] > 0),
            "2 expand: realized growth minus rest, CI > 0": (c2, c2[1] > 0),
            "3 expand: realized downside minus rest <= 0 (point)": (c3, c3[0] <= 0)}
    passed = all(v[1] for v in crit.values())
    tab.to_csv(vp("reports/verdict_v2_eval.csv"), index=False)
    json.dump({"passed": bool(passed), "criteria": {k: {"est": v[0][0], "lo": v[0][1], "hi": v[0][2], "ok": bool(v[1])} for k, v in crit.items()},
               "raw": {k: list(v) for k, v in raw.items()}}, open(vp("reports/verdict_v2_decision.json"), "w"), indent=1, default=float)
    return tab, crit, raw, passed, A_, J_


if __name__ == "__main__":
    tab, crit, raw, passed, A, J = main()
    pd.set_option("display.width", 220)
    print("=== OOF tier table (realized outcomes, raw means with country-clustered 95% CI) ===")
    t = tab.copy()
    t["mean [95% CI]"] = t.apply(lambda x: f"{x.mean_realized:.4f} [{x.lo:.4f}, {x.hi:.4f}]", axis=1)
    print(t[["sample", "tier", "outcome", "n_cells", "mean [95% CI]"]].to_string(index=False))
    print("\n=== Registered criteria (outcomes demeaned within industry x as-of year; tier minus rest) ===")
    for k, (e, ok) in crit.items():
        print(f"{k}: {e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}]  -> {'PASS' if ok else 'FAIL'}")
    print("raw (not demeaned):", {k: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for k, v in raw.items()})
    print("\nPASS CRITERION:", "PASSED - ship v2 labels" if passed else "FAILED - ship rankings + intervals, NO verdict labels (unranked)")
    print("\nTier counts per industry (joint sample, as-of 2023-2024):"); print(pd.crosstab(J.industry, J.tier).to_string())
    print("Tier counts per industry (downside sample, as-of 2022-2024):"); print(pd.crosstab(A.industry, A.tier).to_string())

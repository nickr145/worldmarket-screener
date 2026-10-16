"""Old (shipped-v1) vs new (shipped-v1.1) comparison, computed from the files. python -m screener.score.compare_versions -> reports/comparison_v1_v1_1.md"""
import json

import numpy as np
import pandas as pd

from screener.features.run_panel import ROOT

BEGIN, END = "<!-- v1-v1.1:begin -->", "<!-- v1-v1.1:end -->"


def load(name, new):
    stem, ext = name.rsplit(".", 1)
    return pd.read_csv(ROOT / f"{stem}{'_v1_1' if new else ''}.{ext}")


def table(name, keys, cols, fmt="{:.4f}"):
    a, b = load(name, False), load(name, True)
    m = a.merge(b, on=keys, suffixes=(" v1", " v1.1"))
    out = m[keys].copy()
    for c in cols:
        out[c + " v1"], out[c + " v1.1"] = m[c + " v1"].map(fmt.format), m[c + " v1.1"].map(fmt.format)
        out[c + " diff"] = (m[c + " v1.1"] - m[c + " v1"]).map("{:+.4f}".format)
    return out, m


def md(df):
    rows = [list(map(str, df.columns))] + [[str(v).replace("|", "/") for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(r) + " |" for r in rows[:1]] + ["|" + "---|" * len(rows[0])] + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def main():
    parts = ["# shipped-v1 vs shipped-v1.1 (computed; v1 = committed files, v1.1 = rerun with the single HCLTECH.NS currency override)\n"]
    t, m = table("reports/baselines.csv", ["spec", "cv", "model"], ["rmse", "r2", "spearman"])
    t = t[t.spec.str.startswith("PRIMARY") & (t.cv == "country")]
    parts += ["## Baselines (primary spec, country CV)", md(t), ""]
    t, _ = table("reports/boosted.csv", ["cv", "model"], ["rmse", "r2", "spearman"])
    parts += ["## Boosted-model comparison (growth, country CV)", md(t[t.cv == "country"]), ""]
    t, _ = table("reports/promotion_tests.csv", ["cv", "model", "vs", "metric"], ["diff_model_better", "lo", "hi"])
    parts += ["## Boosted-model promotion tests (country CV; none favours the boosted model in either version)", md(t[t.cv == "country"]), ""]
    t, _ = table("reports/risk_targets.csv", ["set", "target", "cv", "model"], ["rmse", "r2", "spearman"])
    parts += ["## Risk targets (confirmatory, country CV)", md(t[t.set.str.startswith("confirmatory") & (t.cv == "country")]), ""]
    t, _ = table("reports/verdict_v2_eval.csv", ["sample", "tier", "outcome"], ["mean_realized", "lo", "hi"])
    d0, d1 = (json.load(open(ROOT / f"reports/verdict_v2_decision{s}.json")) for s in ("", "_v1_1"))
    crit = pd.DataFrame([{"criterion": k, "v1 est [lo, hi]": f"{d0['criteria'][k]['est']:+.4f} [{d0['criteria'][k]['lo']:+.4f}, {d0['criteria'][k]['hi']:+.4f}]",
                          "v1.1 est [lo, hi]": f"{v['est']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]", "passed v1": d0["criteria"][k]["ok"], "passed v1.1": v["ok"]} for k, v in d1["criteria"].items()])
    parts += [f"## Verdict-tier evidence (registered v2 criterion: passed v1 = {d0['passed']}, v1.1 = {d1['passed']})", md(crit), "", md(t), ""]
    a, b = load("reports/scored_cells.csv", False), load("reports/scored_cells.csv", True)
    k = ["country", "industry"]
    cov = pd.concat([a.groupby("industry").industry_coverage_loco.first().rename("coverage v1"), b.groupby("industry").industry_coverage_loco.first().rename("coverage v1.1"),
                     a.groupby("industry").model_estimate_halfwidth.first().rename("half-width v1"), b.groupby("industry").model_estimate_halfwidth.first().rename("half-width v1.1")], axis=1).round(4)
    parts += ["## Conformal coverage and half-width per industry (leave-country-out, nominal 0.80)", md(cov.reset_index()), ""]
    vc = pd.concat([a.groupby(["industry", "verdict"]).size().unstack(fill_value=0).add_suffix(" v1"), b.groupby(["industry", "verdict"]).size().unstack(fill_value=0).add_suffix(" v1.1")], axis=1)
    parts += ["## Verdict counts", md(vc.reset_index()), ""]
    m = a.merge(b, on=k, suffixes=("_a", "_b"))
    rows = []
    for c in [c for c in a.columns if c not in k and c in b.columns]:
        x, y = m[c + "_a"], m[c + "_b"]
        if x.dtype.kind in "fi" and y.dtype.kind in "fi":
            ch = ~np.isclose(x.astype(float), y.astype(float), rtol=0, atol=1e-12, equal_nan=True)
        else:
            ch = ~((x == y) | (x.isna() & y.isna()))
        for i in m.index[ch]:
            rows.append((m.country[i], m.industry[i], c, x[i], y[i], (y[i] - x[i]) if x.dtype.kind in "fi" else ""))
    ch = pd.DataFrame(rows, columns=["country", "industry", "column", "v1", "v1.1", "diff"])
    cells = ch.groupby(k).column.agg(lambda s: ", ".join(sorted(s))).reset_index().rename(columns={"column": "columns changed"})
    parts += [f"## Scored cells with any changed value: {len(cells)} of {len(a)}", "By industry: " + str(cells.industry.value_counts().to_dict()),
              "Verdict changes: " + str(int((m.verdict_a != m.verdict_b).sum())),
              "Largest absolute change per column (excluding identifiers):", md(ch.assign(a=pd.to_numeric(ch["diff"], errors="coerce").abs()).groupby("column").a.max().sort_values(ascending=False).head(15).rename("max |diff|").reset_index()), "",
              md(cells), ""]
    ch.to_csv(ROOT / "reports/scored_cells_changes_v1_v1_1.csv", index=False)
    num = lambda col: pd.to_numeric(ch[ch.column == col]["diff"], errors="coerce").abs().max()
    inr_a, inr_b = (x[(x.country == "IN") & (x.industry == "software")].iloc[0] for x in (a, b))
    c1, c0 = d1["criteria"], d0["criteria"]
    block = "\n".join([
        BEGIN,
        f"Computed by `python -m screener.score.compare_versions` from the v1 files and the `_v1_1` files (full tables: `reports/comparison_v1_v1_1.md`).",
        "",
        f"- **Scored cells with any changed value:** {len(cells)} of {len(a)} ({', '.join(f'{n} {i}' for i, n in cells.industry.value_counts().items())}); verdict changes: {int((m.verdict_a != m.verdict_b).sum())}; "
        f"largest change in a growth estimate {num('model_estimate') * 100:.3f} percentage points.",
        f"- **India software:** top-firm share of revenue {inr_a.top1_share:.1%} -> {inr_b.top1_share:.1%}; concentrated flag {bool(inr_a.concentrated)} -> {bool(inr_b.concentrated)}.",
        "- **Verdict counts (avoid / consider / expand / insufficient data):** " + "; ".join(f"{r.industry} {r['avoid v1']}/{r['consider v1']}/{r['expand v1']}/{r['insufficient_data v1']} -> {r['avoid v1.1']}/{r['consider v1.1']}/{r['expand v1.1']}/{r['insufficient_data v1.1']}" for _, r in vc.reset_index().iterrows()) + ".",
        f"- **Registered v2 criterion:** passed in v1 ({d0['passed']}) and v1.1 ({d1['passed']}); estimates " + "; ".join(f"{k[:1]}: {c0[k]['est']:+.4f} -> {c1[k]['est']:+.4f}" for k in c1) + ".",
        "- **Boosted models:** not promoted in either version (no paired-difference interval favours a boosted model on growth).",
        "- **Conformal coverage by industry (v1 -> v1.1):** " + "; ".join(f"{i} {r['coverage v1']:.3f} -> {r['coverage v1.1']:.3f}" for i, r in cov.iterrows()) + ".",
        END])
    mc = ROOT / "docs/model_card.md"
    t = mc.read_text()
    t = t[:t.index(BEGIN)] + block + t[t.index(END) + len(END):]
    mc.write_text(t)
    (ROOT / "reports/comparison_v1_v1_1.md").write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()

"""Model card page: leads with the registered protocol."""
import pandas as pd
import streamlit as st

from screener.app.services import md_section
from screener.features.run_panel import ROOT, vp
from screener.score.labels import label_table


def fmt(df, cols):
    out = df.copy()
    for m, name in cols:
        out[name] = df.apply(lambda r: "n/a" if pd.isna(r[m]) else f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]", axis=1)
    return out


def render():
    st.header("Model card")
    plan = ROOT / "PLAN.md"
    st.subheader("1. The pre-registered protocol")
    st.caption("Registered before any model was run; later changes are logged as deviations.")
    st.markdown("\n".join(md_section(plan, "Evaluation protocol").splitlines()[1:]).strip())
    st.subheader("2. First verdict rule (v1): every cell came out 'consider', and why")
    st.markdown(md_section(plan, "Verdict rule v1").replace("## ", "### "))
    st.subheader("3. Baselines (primary spec, country-grouped CV)")
    b = pd.read_csv(vp("reports/baselines.csv"))
    b = b[b.spec.str.startswith("PRIMARY") & (b.cv == "country")]
    names = {"global_mean": "Global mean", "industry_mean": "Industry mean", "trailing_persistence": "Trailing-growth persistence", "ridge_macro": "Ridge, macro only", "ridge_full": "Ridge with trailing growth"}
    t = fmt(b.assign(model=b.model.map(names)), [("rmse", "RMSE [95% bootstrap interval]"), ("r2", "R2 [95% bootstrap interval]"), ("spearman", "Spearman [95% bootstrap interval]"), ("spread", "Top-minus-bottom third [95% bootstrap interval]")])
    st.dataframe(t[["model", "n_rows", "RMSE [95% bootstrap interval]", "R2 [95% bootstrap interval]", "Spearman [95% bootstrap interval]", "Top-minus-bottom third [95% bootstrap interval]"]].rename(columns={"model": "Model", "n_rows": "Rows"}),
                 hide_index=True, width="stretch", key="baseline_table")
    st.caption("Spearman and spread are n/a for constant predictors. Intervals are country-clustered bootstrap.")
    st.subheader("4. Boosted models: not promoted")
    p = pd.read_csv(vp("reports/promotion_tests.csv"))
    p = p[p.cv == "country"]
    p["Difference (positive = boosted better) [95% bootstrap interval]"] = p.apply(lambda r: f"{r.diff_model_better:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", axis=1)
    st.dataframe(p[["model", "vs", "metric", "Difference (positive = boosted better) [95% bootstrap interval]"]].rename(columns={"model": "Boosted model", "vs": "Versus", "metric": "Metric"}),
                 hide_index=True, width="stretch", key="boosted_table")
    st.markdown("Every interval includes zero, so by the registered rule the baselines are the shipped models.")
    st.subheader("5. Risk targets (downside deviation and max drawdown, 12 months, as-of 2022-2024)")
    r = pd.read_csv(vp("reports/risk_targets.csv"))
    r = r[r.set.str.startswith("confirmatory") & (r.cv == "country")]
    t = fmt(r, [("rmse", "RMSE [95% bootstrap interval]"), ("r2", "R2 [95% bootstrap interval]"), ("spearman", "Spearman [95% bootstrap interval]")])
    st.dataframe(t[["target", "model", "RMSE [95% bootstrap interval]", "R2 [95% bootstrap interval]", "Spearman [95% bootstrap interval]"]].rename(columns={"target": "Target", "model": "Model"}),
                 hide_index=True, width="stretch", key="risk_table")
    st.markdown("Trailing downside-deviation persistence is the best model for downside volatility (a stronger signal than growth); nothing beats the industry mean on max drawdown, which is shown as an ordering only. No boosted model was promoted.")
    st.subheader("6. The second verdict rule (v2) is a post-hoc revision")
    v = md_section(plan, "Verdict rule v1")
    st.markdown(v[v.index("**Verdict rule v2"):] if "**Verdict rule v2" in v else v)
    st.markdown(md_section(plan, "Freeze").replace("## ", "### "))
    st.subheader("7. Label names")
    st.caption("The UI uses the display labels. The registered sections above, the data files and the tests use the internal tier names.")
    st.dataframe(label_table(), hide_index=True, width="stretch", key="label_map_card")
    st.subheader("8. v1 and v1.1")
    st.markdown(md_section(ROOT / "docs/model_card.md", "v1 and v1.1").replace("## ", "### "))
    st.subheader("9. Limitations")
    st.markdown(md_section(ROOT / "docs/model_card.md", "Limitations").replace("## ", "### ") or md_section(ROOT / "docs/model_card.md", "Limit"))
    st.markdown(md_section(plan, "Scope limits").replace("## ", "### "))

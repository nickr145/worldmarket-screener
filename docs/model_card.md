# Model card - shipped screener models (frozen as `shipped-v1.1`; the first run, `shipped-v1`, is kept and compared below)

**What it is.** A low-skill screening tool. For each country x industry cell of listed companies it reports recent growth and recent downside volatility, ranks the cell within its industry, and assigns a label. It is not a forecast of a new company's success and not advice.

**Order of presentation (lead with this):** 1) the pre-registered protocol (PLAN.md, "Evaluation protocol"); 2) baseline results; 3) the boosted-model result (not promoted); 4) conformal coverage; 5) limitations below.

**Shipped signals.** Growth ranking = trailing real revenue growth persistence (out-of-fold Spearman about 0.53). Growth point estimate = industry mean + 0.5 x (ridge with trailing growth - industry mean), 80% conformal interval (industry-specific; leave-country-out coverage 0.79-0.84). Downside signal = trailing 12-month downside deviation persistence (R2 about 0.36, Spearman about 0.67; stronger than the growth signal). Max drawdown: ordering only. Boosted models were tested once and not promoted.

**Labels (plain language).** The UI shows descriptive display labels; the internal tier names are frozen in the data files, tests and the registered sections.

| Display label | Internal tier name | Rule (within each industry) |
|---|---|---|
| Strong momentum, volatility not elevated | expand | Top third of the industry on recent growth, and not in the top third on recent downside volatility |
| Neutral | consider | Not in the top third for recent growth and not in the top third for downside volatility; this includes markets with weak recent growth |
| Elevated volatility | avoid | Top third of the industry on recent downside volatility |
| Not rated (reference only) | consider | Turkey is scored for reference only (excluded from the model's training data); its internal tier is 'consider' but it is shown as Not rated |

The limitations below use the internal names.

*Deviation, 2026-10-14: middle label renamed and Turkey display changed (display text only; rule, tiers and outputs unchanged).* The `consider` display label was "No strong signal" and is now "Neutral": "not in the top third for recent growth and not in the top third for downside volatility; this includes markets with weak recent growth" (the old text wrongly implied that weak growth was not a signal). Turkey (internal tier `consider`) is displayed as "Not rated (reference only)", and its growth is described as unadjusted for inflation (not CPI-deflated), because it is not deflated in the data. A display-only flag "thin cell: fewer than 10 firms" (‡) was added to the flag legend; it does not change any label or the table order. Tiers are thirds of the within-industry ranking, so two markets on either side of a cutoff (for example downside rank 16 vs 17 of 25) are not meaningfully different. Conformal half-widths are constant within an industry, so the displayed intervals do not distinguish more uncertain cells from less uncertain ones.

*Deviation, 2026-10-14: label renamed. The `expand` display label was "Strong momentum, lower volatility" and is now "Strong momentum, volatility not elevated" (display text only; rule, tiers and outputs unchanged).*

*Deviation, 2026-10-15: currency-tag correction and rerun (shipped-v1 -> shipped-v1.1).* HCLTECH.NS was tagged USD while its revenue is in INR, so it was converted at FX 1.0 and deflated by US CPI; it made up 94.5% of India software's top-25 revenue. One override (HCLTECH.NS to INR; 4 of 4 size metrics agree, 87x FX gap, tag differs from listing currency, confirmed by HCLTech's FY26 results release: revenue Rs 130,144 crore, $14.7 billion) is now applied from `data/panel/currency_overrides.csv`. A tag-free anomaly check runs on every panel build and prints candidates; it is report-only and never changes data. The pipeline was rerun once under the registered protocol with no new grids, features or thresholds, after a control run with an empty override table that reproduced shipped-v1 exactly. **The corrected data is adopted regardless of how the results come out.** The v1 results remain in `models/shipped_v1/` and `reports/scored_cells.csv`; the v1.1 results are the `_v1_1` files and `models/shipped_v1_1/`.

*Technical corrections to the rerun (no result-driven changes).* (a) The first v1.1 run was invalid because one panel read (in `baselines.py`) still pointed at the v1 panel; it was noticed because only 1 scored cell changed instead of the expected 27, fixed, and the whole chain rerun. A later review found `score_cells` also read an unversioned `gate2_cells.csv` (same `usable` flags, so scored cells are unchanged); a test now fails if any module reads or writes an untagged generated file. (b) The committed v1 `baselines.csv` has 10 USD-target sensitivity rows (spec S3) that the v1 code does not reproduce even with an empty override table; the v1.1 values are the reproducible ones and v1's S3 rows should be read as stale. S3 does not enter scoring. (c) Turkey narrations often fall back to the deterministic template because the narrator's checks reject inflation and "due to" wording; this is expected for a low-confidence country. `selection_sensitivity_v1_1.csv` was rebuilt: only India software (2022, 2023) changes because of the override; eight healthcare rows of the old file were already stale because it was written before the financials fetch finished (46 healthcare firms were cached afterwards; reproduced exactly from the earlier cache state).

## v1 and v1.1
<!-- v1-v1.1:begin -->
Computed by `python -m screener.score.compare_versions` from the v1 files and the `_v1_1` files (full tables: `reports/comparison_v1_v1_1.md`).

- **Scored cells with any changed value:** 27 of 100 (27 software); verdict changes: 0; largest change in a growth estimate 0.027 percentage points.
- **India software:** top-firm share of revenue 94.5% -> 22.4%; concentrated flag True -> False.
- **Verdict counts (avoid / consider / expand / insufficient data):** healthcare 9/12/5/1 -> 9/12/5/1; non_bank_financials 7/9/4/1 -> 7/9/4/1; retail 8/11/6/0 -> 8/11/6/0; software 8/12/6/1 -> 8/12/6/1.
- **Registered v2 criterion:** passed in v1 (True) and v1.1 (True); estimates 1: +0.0407 -> +0.0407; 2: +0.0464 -> +0.0464; 3: -0.0168 -> -0.0168.
- **Boosted models:** not promoted in either version (no paired-difference interval favours a boosted model on growth).
- **Conformal coverage by industry (v1 -> v1.1):** healthcare 0.820 -> 0.820; non_bank_financials 0.794 -> 0.794; retail 0.808 -> 0.808; software 0.840 -> 0.840.
<!-- v1-v1.1:end -->

## Limitations
1. **v2 is a post-hoc revision.** It was registered after seeing v1's direction-only counts (20 expand / 21 avoid under direction alone), although the thresholds, the pass criterion and the evaluation were fixed before the v2 evaluation was run.
2. **Multiple-comparison caution.** Many specifications have been run on this one dataset (growth and risk targets, two horizons, eight sensitivities, two boosted models, two verdict rules). The registered tests limit but do not remove the chance that a favourable result is partly luck.
3. **Criterion 3 is partly mechanical:** `expand` excludes the top downside third by construction, so its lower realized downside is partly built in.
4. **`expand` growth evidence rests on two as-of years (2023 and 2024)**, 40 cell-years from 18 countries (the 170 cell-years they are compared with span 28), in one post-pandemic inflation/rate regime.
5. **Decision-date timing:** for the live decision date (30 Sep 2026), most of the h=1 target year (FY2026) has already elapsed for December filers, so the labels describe **recent momentum and volatility, not a fresh-year forecast**.
6. **h=2 was not tested for verdict labels.** Only h=1 growth and 12-month downside deviation were evaluated.
7. Non-bank financials is the least reliable industry (widest intervals, widest bootstrap range on coverage, fewest cells); the UI shows a `lower_reliability` flag.

8. **Provenance of the manifest hash.** The `scored_cells` input hash in `models/shipped_v1/manifest.json` is the freeze-time value; display columns were added to `scored_cells.csv` after the freeze without changing any model output or verdict (verified by regenerating `scored_cells.csv` with an identical sha256).

9. **Missing FX.** Firm-years with revenue but no FX rate (11 Turkish, 3 Korean) are dropped from membership (found in the 2026-10-15 currency audit; not fixed).
10. **Currency tags near the listing currency cannot be verified by magnitude.** Where the tag and listing currencies are within about 3x of each other (for example CNY or USD on a HKD, CAD or GBP listing), the tag cannot be checked from the size of the numbers, which bounds the error at about 1.5x. Eight firms near their cell's top-25 cutoff are in that position: 0162.HK, 8329.HK, 2348.HK, 8049.HK, 8205.HK, CPH.TO, HLS.TO, DUG.AX.

**Column note.** `verdict_loose_no_interval_rule` in `reports/scored_cells*.csv` is a historical column and should not be used. It was introduced to hold the direction-only verdict (the v1 rule without the interval test: 20 expand / 56 consider / 21 avoid among the 100 cells, 3 insufficient data), but when v2 was registered it was reassigned to a copy of `verdict_v1` taken before the insufficient-data overwrite, so it reads `consider` in all 100 rows. The direction-only counts are recomputed from `recent_growth_signal` and `recent_downside_signal` in `test_docs_numbers.py`. No data dictionary exists beyond this line.

Other limits: listed-firm growth reflects global revenue, not home-market demand; blue-ocean counts are a current snapshot; 2022-2025 fiscal years only; survivorship (current listings, four consecutive fiscal years required); Korea healthcare is not thinly sampled (57 of 60 fetched firms have four fiscal years of revenue and CPI; an earlier figure of 32 matches the count before the fetch finished), but Korea's unresolved domiciles are assumed domestic and flagged; two configured countries are not in the panel (Taiwan, which has no structural macro data, and Argentina, which has fewer than 5 domestic listed firms in every industry) and eight further candidates (AE, CO, EG, PH, VN, IE, NG, KE) have no screener coverage; Turkey scored with low confidence and never actionable.

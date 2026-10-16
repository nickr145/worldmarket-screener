# shipped-v1 vs shipped-v1.1 (computed; v1 = committed files, v1.1 = rerun with the single HCLTECH.NS currency override)

## Baselines (primary spec, country CV)
| spec | cv | model | rmse v1 | rmse v1.1 | rmse diff | r2 v1 | r2 v1.1 | r2 diff | spearman v1 | spearman v1.1 | spearman diff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PRIMARY h1 real / trailing / no-TR / no-biotech | country | global_mean | 0.0664 | 0.0664 | +0.0000 | -0.0272 | -0.0272 | -0.0000 | nan | nan | +nan |
| PRIMARY h1 real / trailing / no-TR / no-biotech | country | industry_mean | 0.0621 | 0.0622 | +0.0000 | 0.1007 | 0.1007 | -0.0000 | nan | nan | +nan |
| PRIMARY h1 real / trailing / no-TR / no-biotech | country | trailing_persistence | 0.0674 | 0.0674 | +0.0000 | -0.0584 | -0.0591 | -0.0007 | 0.5275 | 0.5273 | -0.0002 |
| PRIMARY h1 real / trailing / no-TR / no-biotech | country | ridge_macro | 0.0627 | 0.0628 | +0.0000 | 0.0829 | 0.0828 | -0.0000 | 0.1317 | 0.1321 | +0.0004 |
| PRIMARY h1 real / trailing / no-TR / no-biotech | country | ridge_full | 0.0596 | 0.0596 | +0.0000 | 0.1733 | 0.1723 | -0.0009 | 0.4133 | 0.4132 | -0.0001 |

## Boosted-model comparison (growth, country CV)
| cv | model | rmse v1 | rmse v1.1 | rmse diff | r2 v1 | r2 v1.1 | r2 diff | spearman v1 | spearman v1.1 | spearman diff |
|---|---|---|---|---|---|---|---|---|---|---|
| country | global_mean | 0.0664 | 0.0664 | +0.0000 | -0.0272 | -0.0272 | -0.0000 | nan | nan | +nan |
| country | industry_mean | 0.0621 | 0.0622 | +0.0000 | 0.1007 | 0.1007 | -0.0000 | nan | nan | +nan |
| country | trailing_persistence | 0.0674 | 0.0674 | +0.0000 | -0.0584 | -0.0591 | -0.0007 | 0.5275 | 0.5273 | -0.0002 |
| country | ridge_macro | 0.0627 | 0.0628 | +0.0000 | 0.0829 | 0.0828 | -0.0000 | 0.1317 | 0.1321 | +0.0004 |
| country | ridge_full | 0.0596 | 0.0596 | +0.0000 | 0.1733 | 0.1723 | -0.0009 | 0.4133 | 0.4132 | -0.0001 |
| country | xgboost | 0.0601 | 0.0602 | +0.0001 | 0.1593 | 0.1572 | -0.0020 | 0.4317 | 0.4294 | -0.0024 |
| country | lightgbm | 0.0582 | 0.0583 | +0.0001 | 0.2101 | 0.2090 | -0.0010 | 0.4478 | 0.4511 | +0.0032 |

## Boosted-model promotion tests (country CV; none favours the boosted model in either version)
| cv | model | vs | metric | diff_model_better v1 | diff_model_better v1.1 | diff_model_better diff | lo v1 | lo v1.1 | lo diff | hi v1 | hi v1.1 | hi diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| country | xgboost | ridge_full | rmse | -0.0005 | -0.0005 | -0.0000 | -0.0035 | -0.0034 | +0.0000 | 0.0021 | 0.0021 | -0.0000 |
| country | xgboost | trailing_persistence | spearman | -0.0958 | -0.0979 | -0.0022 | -0.2264 | -0.2268 | -0.0005 | 0.0193 | 0.0184 | -0.0010 |
| country | lightgbm | ridge_full | rmse | 0.0013 | 0.0013 | -0.0000 | -0.0012 | -0.0011 | +0.0001 | 0.0035 | 0.0034 | -0.0001 |
| country | lightgbm | trailing_persistence | spearman | -0.0796 | -0.0762 | +0.0034 | -0.2214 | -0.2171 | +0.0043 | 0.0396 | 0.0427 | +0.0031 |

## Risk targets (confirmatory, country CV)
| set | target | cv | model | rmse v1 | rmse v1.1 | rmse diff | r2 v1 | r2 v1.1 | r2 diff | spearman v1 | spearman v1.1 | spearman diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| confirmatory 2022-2024 | dd_dev_12m | country | global_mean | 0.0418 | 0.0418 | +0.0000 | -0.0328 | -0.0328 | +0.0000 | nan | nan | +nan |
| confirmatory 2022-2024 | dd_dev_12m | country | industry_mean | 0.0419 | 0.0419 | +0.0000 | -0.0389 | -0.0389 | +0.0000 | nan | nan | +nan |
| confirmatory 2022-2024 | dd_dev_12m | country | trailing_persistence | 0.0330 | 0.0330 | +0.0000 | 0.3578 | 0.3578 | +0.0000 | 0.6733 | 0.6733 | +0.0000 |
| confirmatory 2022-2024 | dd_dev_12m | country | ridge_macro | 0.0432 | 0.0432 | +0.0000 | -0.1009 | -0.1009 | +0.0000 | -0.0094 | -0.0094 | +0.0000 |
| confirmatory 2022-2024 | dd_dev_12m | country | ridge_trailing_measure | 0.0339 | 0.0339 | +0.0000 | 0.3221 | 0.3221 | +0.0000 | 0.5919 | 0.5919 | +0.0000 |
| confirmatory 2022-2024 | dd_dev_12m | country | xgboost | 0.0372 | 0.0372 | +0.0000 | 0.1838 | 0.1838 | +0.0000 | 0.4897 | 0.4897 | +0.0000 |
| confirmatory 2022-2024 | dd_dev_12m | country | lightgbm | 0.0364 | 0.0364 | +0.0000 | 0.2154 | 0.2154 | +0.0000 | 0.5418 | 0.5418 | +0.0000 |
| confirmatory 2022-2024 | mdd_12m | country | global_mean | 0.0801 | 0.0801 | +0.0000 | -0.0324 | -0.0324 | +0.0000 | nan | nan | +nan |
| confirmatory 2022-2024 | mdd_12m | country | industry_mean | 0.0796 | 0.0796 | +0.0000 | -0.0199 | -0.0199 | +0.0000 | nan | nan | +nan |
| confirmatory 2022-2024 | mdd_12m | country | trailing_persistence | 0.0858 | 0.0858 | +0.0000 | -0.1834 | -0.1834 | +0.0000 | 0.3615 | 0.3615 | +0.0000 |
| confirmatory 2022-2024 | mdd_12m | country | ridge_macro | 0.0796 | 0.0796 | +0.0000 | -0.0199 | -0.0199 | +0.0000 | 0.0895 | 0.0895 | +0.0000 |
| confirmatory 2022-2024 | mdd_12m | country | ridge_trailing_measure | 0.0770 | 0.0770 | +0.0000 | 0.0473 | 0.0473 | +0.0000 | 0.1532 | 0.1532 | +0.0000 |
| confirmatory 2022-2024 | mdd_12m | country | xgboost | 0.0787 | 0.0787 | +0.0000 | 0.0033 | 0.0033 | +0.0000 | 0.1066 | 0.1066 | +0.0000 |
| confirmatory 2022-2024 | mdd_12m | country | lightgbm | 0.0787 | 0.0787 | +0.0000 | 0.0041 | 0.0041 | +0.0000 | 0.0751 | 0.0751 | +0.0000 |

## Verdict-tier evidence (registered v2 criterion: passed v1 = True, v1.1 = True)
| criterion | v1 est [lo, hi] | v1.1 est [lo, hi] | passed v1 | passed v1.1 |
|---|---|---|---|---|
| 1 avoid: realized downside minus rest, CI > 0 | +0.0407 [+0.0255, +0.0545] | +0.0407 [+0.0255, +0.0545] | True | True |
| 2 expand: realized growth minus rest, CI > 0 | +0.0464 [+0.0232, +0.0705] | +0.0464 [+0.0231, +0.0705] | True | True |
| 3 expand: realized downside minus rest <= 0 (point) | -0.0168 [-0.0307, -0.0017] | -0.0168 [-0.0307, -0.0017] | True | True |

| sample | tier | outcome | mean_realized v1 | mean_realized v1.1 | mean_realized diff | lo v1 | lo v1.1 | lo diff | hi v1 | hi v1.1 | hi diff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (2022-2024, downside) | avoid | dd_dev_12m | 0.1582 | 0.1582 | +0.0000 | 0.1458 | 0.1458 | +0.0000 | 0.1693 | 0.1693 | +0.0000 |
| A (2022-2024, downside) | expand | dd_dev_12m | 0.1222 | 0.1222 | +0.0000 | 0.1082 | 0.1082 | +0.0000 | 0.1400 | 0.1400 | +0.0000 |
| A (2022-2024, downside) | consider | dd_dev_12m | 0.1163 | 0.1163 | +0.0000 | 0.1074 | 0.1074 | +0.0000 | 0.1264 | 0.1264 | +0.0000 |
| J (2023-2024, joint) | avoid | dd_dev_12m | 0.1622 | 0.1622 | +0.0000 | 0.1471 | 0.1471 | +0.0000 | 0.1747 | 0.1747 | +0.0000 |
| J (2023-2024, joint) | avoid | target_h1 | 0.0378 | 0.0378 | +0.0001 | 0.0218 | 0.0218 | +0.0000 | 0.0606 | 0.0607 | +0.0001 |
| J (2023-2024, joint) | expand | dd_dev_12m | 0.1222 | 0.1222 | +0.0000 | 0.1096 | 0.1096 | +0.0000 | 0.1389 | 0.1389 | +0.0000 |
| J (2023-2024, joint) | expand | target_h1 | 0.0856 | 0.0856 | +0.0000 | 0.0627 | 0.0627 | +0.0000 | 0.1087 | 0.1087 | +0.0000 |
| J (2023-2024, joint) | consider | dd_dev_12m | 0.1207 | 0.1207 | +0.0000 | 0.1103 | 0.1103 | +0.0000 | 0.1322 | 0.1322 | +0.0000 |
| J (2023-2024, joint) | consider | target_h1 | 0.0385 | 0.0384 | -0.0000 | 0.0242 | 0.0242 | -0.0000 | 0.0517 | 0.0517 | -0.0000 |

## Conformal coverage and half-width per industry (leave-country-out, nominal 0.80)
| industry | coverage v1 | coverage v1.1 | half-width v1 | half-width v1.1 |
|---|---|---|---|---|
| healthcare | 0.82 | 0.82 | 0.0486 | 0.0486 |
| non_bank_financials | 0.7941 | 0.7941 | 0.1145 | 0.1145 |
| retail | 0.8085 | 0.8085 | 0.0629 | 0.0629 |
| software | 0.84 | 0.84 | 0.0597 | 0.0598 |

## Verdict counts
| industry | avoid v1 | consider v1 | expand v1 | insufficient_data v1 | avoid v1.1 | consider v1.1 | expand v1.1 | insufficient_data v1.1 |
|---|---|---|---|---|---|---|---|---|
| healthcare | 9 | 12 | 5 | 1 | 9 | 12 | 5 | 1 |
| non_bank_financials | 7 | 9 | 4 | 1 | 7 | 9 | 4 | 1 |
| retail | 8 | 11 | 6 | 0 | 8 | 11 | 6 | 0 |
| software | 8 | 12 | 6 | 1 | 8 | 12 | 6 | 1 |

## Scored cells with any changed value: 27 of 100
By industry: {'software': 27}
Verdict changes: 0
Largest absolute change per column (excluding identifiers):
| column | max |diff| |
|---|---|
| top1_share | 0.721669376348506 |
| median_growth_retained | 0.002562948202720499 |
| model_estimate_lo | 0.0003335802921774905 |
| model_estimate_hi | 0.00028404876191401307 |
| model_estimate | 0.0002720296613956952 |
| v1_growth_threshold | 0.0002224981311321969 |
| model_estimate_halfwidth | 6.155063078189943e-05 |
| concentrated | nan |
| low_confidence_flag | nan |

| country | industry | columns changed |
|---|---|---|
| AU | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| BR | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| CA | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| CN | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| DE | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| ES | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| FR | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| GB | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| HK | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| ID | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| IL | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| IN | software | concentrated, low_confidence_flag, median_growth_retained, model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, top1_share, v1_growth_threshold |
| IT | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| JP | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| KR | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| MY | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| NL | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| NO | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| NZ | software | model_estimate_halfwidth, v1_growth_threshold |
| PL | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| SA | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| SE | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| SG | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| TH | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| TR | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| US | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |
| ZA | software | model_estimate, model_estimate_halfwidth, model_estimate_hi, model_estimate_lo, v1_growth_threshold |

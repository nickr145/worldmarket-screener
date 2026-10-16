import pandas as pd
import yaml
from pathlib import Path

from screener.clean.fiscal import as_of_date, fiscal_year, known_at

CFG = Path(__file__).resolve().parents[1] / "config"


def test_fiscal_year_rule():
    assert fiscal_year("2024-03-31") == 2023
    assert fiscal_year("2024-06-30") == 2024
    assert fiscal_year("2024-12-31") == 2024


def test_reporting_lag_blocks_leakage():
    # FY2023 (Dec FYE) is NOT known on the FYE date or 119 days later; known at 120 days
    assert not known_at("2023-12-31", "2023-12-31")
    assert not known_at("2023-12-31", "2024-04-28")
    assert known_at("2023-12-31", "2024-04-29")
    assert as_of_date("2023-12-31") == pd.Timestamp("2024-04-29")


def test_case_countries_in_config():
    case = {"AR", "BR", "CA", "CN", "DE", "GB", "ID", "IN", "JP", "KR", "PL", "SG", "US"}  # the 13 countries of an earlier project; all must stay covered
    have = set(yaml.safe_load(open(CFG / "countries.yaml"))["countries"])
    assert case <= have, f"missing case countries: {case - have}"

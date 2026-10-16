"""The public export must refuse anything under reports/audits/ and must not carry the per-firm evidence column of the override table."""
import importlib.util
import sys

import pytest

from screener.features.run_panel import ROOT

SCRIPT = ROOT / "scripts/export_public.py"
pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="export script is not part of the public export")
if SCRIPT.exists():
    spec = importlib.util.spec_from_file_location("export_public", SCRIPT)
    EX = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(EX)


def test_audits_are_forbidden_pattern():
    assert EX.pat(EX.FORBIDDEN_PATHS)("reports/audits/currency_review.csv")


def test_export_fails_if_audits_would_be_included(tmp_path, monkeypatch):
    (ROOT / "reports/audits").mkdir(exist_ok=True)
    if not list((ROOT / "reports/audits").glob("*.csv")):
        pytest.skip("no audit files in this checkout")
    monkeypatch.setattr(EX, "OUT", tmp_path / "out")
    monkeypatch.setattr(EX, "ALLOW_FILES", EX.ALLOW_FILES + ["reports/audits/*.csv"])
    monkeypatch.setattr(sys, "argv", ["export_public.py"])
    with pytest.raises(AssertionError, match="forbidden"):
        EX.build()
    assert not list((tmp_path / "out").rglob("*.csv"))  # nothing copied before the check


def test_override_table_exported_without_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(EX, "OUT", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["export_public.py"])
    EX.build()
    cols = list(EX.pd.read_csv(tmp_path / "out/data/panel/currency_overrides.csv").columns)
    assert cols == ["symbol", "corrected_ccy", "reason"]
    assert not (tmp_path / "out/reports/audits").exists()

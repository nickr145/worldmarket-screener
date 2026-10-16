"""Import boundary: the model code never imports anthropic; only screener/parse talks to the API."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPORT = re.compile(r"^\s*(import|from)\s+anthropic\b", re.M)


def files(sub):
    return [p for p in (ROOT / sub).rglob("*.py")]


def test_model_never_imports_anthropic():
    offenders = [str(p) for p in files("model") if IMPORT.search(p.read_text())]
    assert not offenders, offenders
    # also catches lazy/indirect use of the SDK name or the API client helper
    assert not [str(p) for p in files("model") if "get_client" in p.read_text() or "screener.parse" in p.read_text()]


def test_only_parse_package_imports_anthropic():
    offenders = [str(p) for p in ROOT.rglob("*.py") if "tests" not in p.parts and "parse" not in p.parts and IMPORT.search(p.read_text())]
    assert not offenders, offenders
    # parse/ imports it lazily inside get_client(), never at module import time of schema/narrate-check logic
    assert not IMPORT.search((ROOT / "parse/schema.py").read_text())

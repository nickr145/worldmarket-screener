"""Live canned-prompt check (hits the API; run with RUN_LIVE=1). 16 prompts incl. banks/insurers (unsupported), an ambiguous one,
an unscored country and a prompt-injection attempt."""
import os

import pytest

from screener.parse.canned import CANNED, run_all

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="live API test; set RUN_LIVE=1")


def test_all_canned_prompts_parse_as_expected():
    rows = run_all()
    bad = [(r["id"], r["issues"]) for r in rows if r["issues"] != "ok"]
    assert len(rows) == len(CANNED) >= 12 and not bad, bad

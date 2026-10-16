"""Universe ingest: paging, currency mapping, per-call timeout, resumable parts, failures logged and never cached, no overwrite. No network."""
import time

import pandas as pd
import pytest

from screener.ingest import universe as U


def quote(sym, cur="GBp", fin="GBP"):
    return {"symbol": sym, "longName": f"{sym} plc", "exchange": "LSE", "marketCap": 1e9, "currency": cur, "financialCurrency": fin}


def test_pull_pages_and_rows_map_subunit_currencies(monkeypatch):
    monkeypatch.setattr(U, "PAGE", 2)
    pages = {0: {"quotes": [quote("A"), quote("B")], "total": 3}, 2: {"quotes": [quote("C", "ILA", "ILS")], "total": 3}}
    monkeypatch.setattr(U.yf, "screen", lambda q, size, offset, sortField, sortAsc: pages[offset])
    q = U.pull("GB", ["LSE"], "Software—Application", 2e7)
    df = U.rows("GB", "software", "Software—Application", q)
    assert list(df.symbol) == ["A", "B", "C"] and list(df.columns) == U.COLS
    assert list(df.ccy) == ["GBP", "GBP", "ILS"] and list(df.fin_ccy) == ["GBP", "GBP", "ILS"]


def test_two_tries_then_failure_is_logged_not_cached(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(U, "FAIL", tmp_path / "fail.csv")
    monkeypatch.setattr(U.time, "sleep", lambda s: None)
    monkeypatch.setattr(U.yf, "screen", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(ConnectionError("429")))
    assert U.fetch_unit(("GB", ["LSE"], "software", "Software—Application"), 2e7, tmp_path) is False
    assert len(calls) == 2 and not list(tmp_path.glob("*.parquet"))
    assert "universe,GB/Software—Application" in (tmp_path / "fail.csv").read_text()


def test_hard_timeout_abandons_a_stuck_call(monkeypatch):
    t0 = time.time()
    with pytest.raises(TimeoutError):
        U.call_with_timeout(lambda: time.sleep(5), 0.2)
    assert time.time() - t0 < 2


def test_success_is_cached_and_skipped_on_rerun(monkeypatch, tmp_path):
    n = []
    monkeypatch.setattr(U.time, "sleep", lambda s: None)
    monkeypatch.setattr(U.yf, "screen", lambda *a, **k: n.append(1) or {"quotes": [quote("A")], "total": 1})
    u = ("GB", ["LSE"], "software", "Software—Application")
    assert U.fetch_unit(u, 2e7, tmp_path) and U.fetch_unit(u, 2e7, tmp_path)
    assert len(n) == 1 and len(pd.read_parquet(next(tmp_path.glob("*.parquet")))) == 1


def test_existing_output_is_not_overwritten_without_force(tmp_path):
    out = tmp_path / "q.parquet"
    out.write_bytes(b"x")
    with pytest.raises(SystemExit, match="--force"):
        U.main(["--out", str(out), "--parts-dir", str(tmp_path / "p")])
    assert out.read_bytes() == b"x"

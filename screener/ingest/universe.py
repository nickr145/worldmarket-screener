"""Universe quotes: every listed firm that the yfinance screener returns for each (country, yfinance industry) unit
-> data/raw/spike_quotes.parquet, the input of screener.ingest.spike_cells. Needs network access; yfinance results change over time.

Same query as the original pull: region == country, primary exchanges only (countries.yaml), industry, market cap > min_mcap_usd, revenue (LTM) > 0.
Hardening: two quick tries per call, a hard wall-clock timeout per call (a stuck call is abandoned, not waited for), one cache file per unit
(data/raw/universe_parts/, resumable: finished units are skipped on rerun), failures -> reports/fetch_failures.csv. The final file is written
only when every unit succeeded, and an existing spike_quotes.parquet is never overwritten without --force.

python -m screener.ingest.universe [--countries SG,NZ] [--out PATH] [--parts-dir DIR] [--force]
"""
import argparse
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf
from yfinance import EquityQuery as Q

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "screener" / "config"
OUT = ROOT / "data" / "raw" / "spike_quotes.parquet"
PARTS = ROOT / "data" / "raw" / "universe_parts"
FAIL = ROOT / "reports" / "fetch_failures.csv"
CCY = {"GBp": "GBP", "ILA": "ILS", "ZAc": "ZAR"}  # sub-unit trading currencies -> the main unit
COLS = ["country", "industry", "yf_industry", "symbol", "name", "exchange", "mcap", "ccy", "fin_ccy"]
CALL_TIMEOUT = 45  # seconds per screener call
PAGE = 250


def call_with_timeout(fn, seconds):
    """Run fn() in a daemon thread; raise TimeoutError if it has not finished in `seconds` (the thread is abandoned)."""
    box = {}
    t = threading.Thread(target=lambda: box.update(**_run(fn)), daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"no response in {seconds}s")
    if "err" in box:
        raise box["err"]
    return box["ok"]


def _run(fn):
    try:
        return {"ok": fn()}
    except Exception as e:  # noqa: BLE001 - re-raised in the caller
        return {"err": e}


def log_fail(key, msg):
    new = not FAIL.exists()
    with open(FAIL, "a") as f:
        if new:
            f.write("kind,key,message\n")
        f.write(f'universe,{key},"{msg}"\n')


def pull(country, exchanges, yf_industry, min_mcap, timeout=CALL_TIMEOUT):
    """All screener quotes for one unit (paged). Raises after two failed tries of any page."""
    q = Q("and", [Q("eq", ["region", country.lower()]), Q("is-in", ["exchange", *exchanges]),
                  Q("eq", ["industry", yf_industry]), Q("gt", ["intradaymarketcap", min_mcap]),
                  Q("gt", ["totalrevenues.lasttwelvemonths", 0])])
    out, off = [], 0
    while True:
        for attempt in range(2):
            try:
                r = call_with_timeout(lambda: yf.screen(q, size=PAGE, offset=off, sortField="intradaymarketcap", sortAsc=False), timeout)
                break
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(2)
        else:
            raise RuntimeError(str(err)[:80].replace('"', "'"))
        out += r["quotes"]
        off += PAGE
        if off >= r.get("total", 0):
            return out


def rows(country, industry, yf_industry, quotes):
    return pd.DataFrame([{"country": country, "industry": industry, "yf_industry": yf_industry, "symbol": q.get("symbol"),
                          "name": q.get("longName") or q.get("shortName"), "exchange": q.get("exchange"), "mcap": q.get("marketCap"),
                          "ccy": CCY.get(q.get("currency"), q.get("currency")), "fin_ccy": CCY.get(q.get("financialCurrency"), q.get("financialCurrency"))}
                         for q in quotes], columns=COLS)


def units(countries_filter=None):
    cfg = yaml.safe_load(open(CFG / "countries.yaml"))["countries"]
    inds = {p.stem: yaml.safe_load(open(p))["yf_industries"] for p in sorted((CFG / "industries").glob("*.yaml"))}  # all, incl. Biotechnology
    for c, meta in cfg.items():
        if countries_filter and c not in countries_filter:
            continue
        for ind, yfs in inds.items():
            for y in yfs:
                yield c, meta["exchanges"], ind, y


def fetch_unit(u, min_mcap, parts):
    c, exch, ind, y = u
    f = parts / f"{c}__{re.sub(r'[^A-Za-z0-9]+', '_', y).strip('_')}.parquet"
    if f.exists():
        return True
    try:
        df = rows(c, ind, y, pull(c, exch, y, min_mcap))
    except Exception as e:  # noqa: BLE001
        log_fail(f"{c}/{y}", str(e)[:80].replace('"', "'"))
        return False
    df.to_parquet(f)
    time.sleep(0.2)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", help="comma-separated ISO2 codes (default: all in countries.yaml)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--parts-dir", type=Path, default=PARTS)
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    a = ap.parse_args(argv)
    if a.out.exists() and not a.force:
        sys.exit(f"{a.out} exists; pass --force to overwrite (the shipped scoring was built from it)")
    min_mcap = yaml.safe_load(open(CFG / "settings.yaml"))["min_mcap_usd"]
    todo = list(units(set(a.countries.split(",")) if a.countries else None))
    a.parts_dir.mkdir(parents=True, exist_ok=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    print(len(todo), "units", flush=True)
    with ThreadPoolExecutor(4) as ex:
        ok = list(ex.map(lambda u: fetch_unit(u, min_mcap, a.parts_dir), todo))
    if not all(ok):
        sys.exit(f"{ok.count(False)} of {len(todo)} units failed (see {FAIL}); rerun to retry them, finished units are cached")
    parts = [pd.read_parquet(a.parts_dir / f"{c}__{re.sub(r'[^A-Za-z0-9]+', '_', y).strip('_')}.parquet") for c, _, _, y in todo]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df.to_parquet(a.out)
    print(f"wrote {len(df)} quotes from {len(todo)} units to {a.out}")
    return df


if __name__ == "__main__":
    main()

"""Daily adjusted-close history (2015+) for the firms kept in the panel -> data/raw/prices/<symbol>.parquet.

Resumable (existing files skipped), two quick tries per ticker with a 15s HTTP timeout, failures -> reports/fetch_failures.csv.
Prices are in LOCAL listing currency (auto_adjust=True: splits and dividends); returns are unit-free.
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yfinance as yf

from screener.features.run_panel import vp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "prices"
FAIL = ROOT / "reports" / "fetch_failures.csv"


def one(sym):
    f = OUT / f"{re.sub(r'[^A-Za-z0-9._-]', '_', sym)}.parquet"
    if f.exists():
        return
    err = "empty history"
    for a in range(2):
        try:
            h = yf.Ticker(sym).history(start="2015-01-01", auto_adjust=True, timeout=15)
            if len(h) > 20:
                h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
                h[["Close", "Volume"]].rename(columns=str.lower).to_parquet(f)
                time.sleep(0.2)
                return
        except Exception as e:
            err = str(e)[:80].replace('"', "'")
        time.sleep(2)
    new = not FAIL.exists()
    with open(FAIL, "a") as fh:
        if new:
            fh.write("kind,key,message\n")
        fh.write(f'prices,{sym},"{err}"\n')


def main():
    obs = pd.read_parquet(vp("data/panel/firm_obs.parquet"))
    syms = sorted(obs.symbol.unique())
    OUT.mkdir(parents=True, exist_ok=True)
    print(len(syms), "tickers", flush=True)
    with ThreadPoolExecutor(4) as ex:
        for k, _ in enumerate(ex.map(one, syms), 1):
            if k % 200 == 0:
                print(k, flush=True)


if __name__ == "__main__":
    main()

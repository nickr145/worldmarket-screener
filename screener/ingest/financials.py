"""Financials ingest: top-N (by market cap) domestic firms of each passing cell -> data/raw/financials/<symbol>.parquet.

One cache file per ticker (resumable: existing files are skipped), throttled, failures -> reports/fetch_failures.csv.
Long format per fiscal year: symbol, fiscal_year_end, fiscal_year, as_of_date, currency, plus revenue / income / balance lines.
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

from screener.clean.fiscal import as_of_date, fiscal_year

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "screener" / "config"
OUT = ROOT / "data" / "raw" / "financials"
FAIL = ROOT / "reports" / "fetch_failures.csv"
INCOME = {"Total Revenue": "revenue", "Operating Income": "operating_income", "Net Income": "net_income"}
BALANCE = {"Total Debt": "total_debt", "Stockholders Equity": "equity", "Total Assets": "total_assets"}


def log_fail(sym, msg):
    new = not FAIL.exists()
    with open(FAIL, "a") as f:
        if new:
            f.write("kind,key,message\n")
        f.write(f'financials,{sym},"{msg}"\n')


def universe(variant=""):
    st = yaml.safe_load(open(CFG / "settings.yaml"))
    fu = pd.read_csv(ROOT / "reports" / f"firm_universe{variant}.csv")
    cc = pd.read_csv(ROOT / "reports" / f"cell_counts{variant}.csv")
    ok = cc[cc.domestic >= st["min_cell_firms"]][["country", "industry"]]
    d = fu[fu.domestic].merge(ok)
    d = d.sort_values("mcap", ascending=False).groupby(["country", "industry"]).head(st["fetch_top_n"])
    return d, st


def fetch_one(row, countries, st):
    sym = row.symbol
    f = OUT / f"{re.sub(r'[^A-Za-z0-9._-]', '_', sym)}.parquet"
    if f.exists():
        return
    lag = countries[row.country].get("filing_lag_days", st["filing_lag_days"])
    for a in range(4):
        try:
            t = yf.Ticker(sym)
            inc = t.get_income_stmt(freq="yearly", pretty=True)
            if inc is None or inc.empty or "Total Revenue" not in inc.index:
                raise ValueError("empty income statement (rate limit or no data)")  # retried, never cached
            bal = t.get_balance_sheet(freq="yearly", pretty=True)
            df = inc.reindex(list(INCOME)).T.rename(columns=INCOME)
            if bal is not None and not bal.empty:
                df = df.join(bal.reindex(list(BALANCE)).T.rename(columns=BALANCE), how="left")
            df.index = pd.to_datetime(df.index)
            df = df.rename_axis("fiscal_year_end").reset_index()
            df["fiscal_year"] = [fiscal_year(d, st["fy_cutoff_month"]) for d in df.fiscal_year_end]
            df["as_of_date"] = [as_of_date(d, lag) for d in df.fiscal_year_end]
            df["symbol"], df["country"], df["industry"] = sym, row.country, row.industry
            df["currency"] = row.fin_ccy
            df.to_parquet(f)
            time.sleep(0.5)
            return
        except Exception as e:
            time.sleep(3 * 2 ** a)
            err = str(e)[:80].replace('"', "'")
    log_fail(sym, f"failed after retries: {err}")  # not cached, so a rerun retries it


def main():
    import os
    d, st = universe(os.environ.get("WM_VARIANT", ""))
    countries = yaml.safe_load(open(CFG / "countries.yaml"))["countries"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(d.itertuples())
    print(len(rows), "tickers", flush=True)
    with ThreadPoolExecutor(3) as ex:
        for k, _ in enumerate(ex.map(lambda r: fetch_one(r, countries, st), rows), 1):
            if k % 100 == 0:
                print(k, flush=True)


if __name__ == "__main__":
    main()

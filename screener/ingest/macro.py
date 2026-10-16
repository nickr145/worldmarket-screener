"""Macro ingest: World Bank WDI/WGI + IMF DataMapper -> data/raw/macro/<source>_<code>.parquet (cached, resumable).

Long format: iso3, year, value, indicator, name, source, as_of_date (= Dec 31 of year + lag_days).
Taiwan is absent from the World Bank; its CPI/inflation comes from the IMF DataMapper (TWN) only.
"""
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "screener" / "config"
OUT = ROOT / "data" / "raw" / "macro"
FAIL = ROOT / "reports" / "fetch_failures.csv"
S = requests.Session()


def get(url, params=None, tries=4):
    for a in range(tries):
        try:
            r = S.get(url, params=params, timeout=60)
            if r.ok:
                return r.json()
        except Exception:
            pass
        time.sleep(2 ** a)


def fail(kind, key, msg):
    new = not FAIL.exists()
    with open(FAIL, "a") as f:
        if new:
            f.write("kind,key,message\n")
        f.write(f'{kind},{key},"{msg}"\n')


def fetch_wb(code, source, isos, y0, y1):
    url = f"https://api.worldbank.org/v2/country/{';'.join(isos)}/indicator/{code}"
    p = {"format": "json", "per_page": 20000, "date": f"{y0}:{y1}"}
    if source == "wgi":
        p["source"] = 3
    j = get(url, p)
    if not j or len(j) < 2 or j[1] is None:
        return None
    return pd.DataFrame([{"iso3": r["countryiso3code"], "year": int(r["date"]), "value": r["value"]} for r in j[1]])


def fetch_imf(code, isos):
    rows = []
    for iso in isos:  # the multi-country URL form is not reliable; one call per country
        j = get(f"https://www.imf.org/external/datamapper/api/v1/{code}/{iso}")
        d = (j or {}).get("values", {}).get(code, {}).get(iso, {})
        rows += [{"iso3": iso, "year": int(y), "value": v} for y, v in d.items()]
        time.sleep(0.2)
    return pd.DataFrame(rows) if rows else None


def main():
    fx_twd()
    cfg = yaml.safe_load(open(CFG / "macro.yaml"))
    isos = [m["iso3"] for m in yaml.safe_load(open(CFG / "countries.yaml"))["countries"].values()]
    wb_isos = [i for i in isos if i != "TWN"] + ["EMU"]  # EMU = euro-area aggregate (EUR deflator)
    OUT.mkdir(parents=True, exist_ok=True)
    for ind in cfg["indicators"]:
        f = OUT / f"{ind['source']}_{ind['code']}.parquet"
        if f.exists():
            continue
        df = fetch_imf(ind["code"], isos) if ind["source"] == "imf" else fetch_wb(ind["code"], ind["source"], wb_isos, cfg["start_year"], cfg["end_year"])
        if df is None:
            fail("macro", ind["code"], "no data returned")
            print("FAIL", ind["code"])
            continue
        df = df.dropna(subset=["value"]).query("@cfg['start_year'] <= year <= @cfg['end_year']").copy()
        df["indicator"], df["name"], df["source"] = ind["code"], ind["name"], ind["source"]
        df["as_of_date"] = pd.to_datetime(df.year.astype(str) + "-12-31") + pd.Timedelta(days=ind["lag_days"])
        df.to_parquet(f)
        print(ind["code"], len(df), df.iso3.nunique(), "countries", flush=True)
        time.sleep(0.5)


def fx_twd():
    """Taiwan is not in the World Bank: annual mean TWD per USD from yfinance, in the PA.NUS.FCRF long format."""
    f = OUT / "yf_TWD_FCRF.parquet"
    if f.exists():
        return
    h = yf.Ticker("TWD=X").history(period="max", interval="1mo")["Close"]
    a = h.groupby(h.index.year).mean()
    df = pd.DataFrame({"iso3": "TWN", "year": a.index, "value": a.values})
    df = df[(df.year >= 2008) & (df.year <= 2025)].copy()
    df["indicator"], df["name"], df["source"] = "PA.NUS.FCRF", "fx_lcu_per_usd", "yf"
    df["as_of_date"] = pd.to_datetime(df.year.astype(str) + "-12-31") + pd.Timedelta(days=60)
    df.to_parquet(f)


if __name__ == "__main__":
    main()

"""Week 1 spike (v2: primary exchanges, name dedupe, domicile==listing country): count listed firms per (country, industry) cell via the yfinance screener.

Raw quotes cached to data/raw/spike_quotes.parquet; rerun reads the cache.
valid = revenue LTM > 0 and mcap >= min_mcap_usd (screener-side proxy; real FS check is Week 2).
strict = valid and trading currency == financial-reporting currency, which drops most
foreign cross-listings/ADRs (e.g. SAP on WSE). Go/no-go uses strict (conservative).
"""
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

CFG = Path(__file__).resolve().parents[1] / "config"
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = RAW / "spike_quotes.parquet"


def inds_cfg_biotech():
    return yaml.safe_load(open(CFG / "industries" / "healthcare.yaml")).get("include_biotechnology", True)


def norm(name):
    n = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    n = re.sub(r"\b(ltd|limited|inc|incorporated|plc|ag|se|co|corp|corporation|sa|nv|llc|company|holdings?|group|class [a-z]|ordinary shares|adr|the)\b", " ", n)
    return " ".join(n.split())


def domicile(symbols):
    """Ticker.info['country'] per symbol, cached incrementally (slow: ~1.5s/call, threaded)."""
    f = RAW / "domicile.parquet"
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["symbol", "domicile"])
    if os.environ.get("RETRY_ERR") == "1":  # failed/rate-limited lookups are only retried on request (slow: backoff)
        have = have[~have.domicile.isin(["ERR", "?"])]
    todo = [s for s in symbols if s not in set(have.symbol)]

    def get(sym):
        for a in range(2):  # empty info is usually permanent (e.g. KOSDAQ names); 2 tries, short backoff
            try:
                c = yf.Ticker(sym).info.get("country")
                if c:
                    time.sleep(0.2)
                    return sym, c
            except Exception:
                pass
            time.sleep(3 * (a + 1))
        return sym, "ERR"
    out = []
    with ThreadPoolExecutor(4) as ex:
        for k, r in enumerate(ex.map(get, todo), 1):
            out.append(r)
            if k % 500 == 0:
                pd.concat([have, pd.DataFrame(out, columns=have.columns)]).to_parquet(f)
                print("domicile", k, "/", len(todo), flush=True)
    have = pd.concat([have, pd.DataFrame(out, columns=have.columns)]).drop_duplicates("symbol")
    have.to_parquet(f)
    print("unresolved domicile:", (have.domicile == "ERR").sum(), flush=True)
    return have.set_index("symbol").domicile


def main():
    cfg = yaml.safe_load(open(CFG / "countries.yaml"))
    countries = cfg["countries"]
    inds = {p.stem: yaml.safe_load(open(p))["yf_industries"] for p in sorted((CFG / "industries").glob("*.yaml"))}
    st = yaml.safe_load(open(CFG / "settings.yaml"))
    df = pd.read_parquet(CACHE)  # raw pulls (cached); universe rules below are applied on top
    df["industry"] = df.industry.replace({"fintech": "non_bank_financials"})
    BIO = os.environ.get("BIOTECH") == "1"  # sensitivity universe: include Biotechnology, outputs get a _biotech suffix
    SUF = "_biotech" if BIO else ""
    if not (BIO or inds_cfg_biotech()):
        df = df[df.yf_industry != "Biotechnology"]
    df = df[df.country.isin(countries)]
    for c, m in countries.items():  # explicit foreign-registry/NVDR-style lines (e.g. TH -R, -F)
        rx = m.get("exclude_ticker_regex")
        if rx:
            df = df[~((df.country == c) & df.symbol.str.contains(rx, regex=True))]
    df = df[[e in countries[c]["exchanges"] for c, e in zip(df.country, df.exchange)]].copy()
    df["key"] = df.name.map(norm)
    raw = df.groupby("country").size()
    df = df.sort_values("mcap", ascending=False).drop_duplicates(["country", "key"])
    dups = (raw - df.groupby("country").size()).rename("dups_removed")
    print("duplicate listings removed per country (primary exchanges, name-normalised; ISIN unavailable in yfinance):")
    print(pd.concat([raw.rename("primary_exch_listings"), dups], axis=1).to_string(), "\n")
    dom = domicile(df.symbol.tolist())
    df["domicile"] = df.symbol.map(dom)
    ok = {c: {m["name"], *m.get("aliases", [])} for c, m in countries.items()}
    df["domestic"] = [d in ok[c] for c, d in zip(df.country, df.domicile)]
    # unresolved lookups: countries flagged assume_domestic_if_unresolved (KR: KSC/KOE are domestic-only boards) keep the firm
    df["domicile_assumed"] = [d == "ERR" and bool(countries[c].get("assume_domestic_if_unresolved")) for c, d in zip(df.country, df.domicile)]
    df["domestic"] = df.domestic | df.domicile_assumed
    df["old_strict"] = df.ccy == df.fin_ccy
    df.to_csv(ROOT / "reports" / f"firm_universe{SUF}.csv", index=False)
    df["new_not_old"] = df.domestic & ~df.old_strict
    g = df.groupby(["country", "industry"])
    t = pd.DataFrame({"firms_after_dedupe": g.size(), "domestic": g.domestic.sum(),
                      "old_strict_pass": g.old_strict.sum(), "old_strict_pass_and_domestic": (df.domestic & df.old_strict).groupby([df.country, df.industry]).sum(),
                      "domestic_but_old_strict_drops": g.new_not_old.sum()})
    grid = pd.MultiIndex.from_product([countries, inds], names=["country", "industry"])
    t = t.reindex(grid).fillna(0).astype(int)
    t.insert(0, "listed_raw_primary_exch", df.groupby(["country", "industry"]).size().reindex(grid).fillna(0).astype(int) + 0)
    t.to_csv(ROOT / "reports" / f"cell_counts{SUF}.csv")
    pd.set_option("display.max_rows", None, "display.width", 250)
    print(t.to_string())
    n = st["min_cell_firms"]
    for col in ["domestic", "old_strict_pass"]:
        print(f"\ncells >= {n} firms by {col}: {(t[col] >= n).sum()}/{len(t)}  per industry {(t[col] >= n).groupby('industry').sum().to_dict()}")
    print("\nTop non-matching domicile strings (check aliases):")
    print(df[~df.domestic].groupby(["country", "domicile"]).size().sort_values(ascending=False).head(25).to_string())
    go = (t.domestic >= n).sum()
    print("GO" if go >= st["go_min_cells"] else "NO-GO", f"{go}/{len(t)} cells (criterion >= {st['go_min_cells']} of 128)")


if __name__ == "__main__":
    main()

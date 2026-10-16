"""Secondary target + downside measures from daily prices (local-currency, dividend/split adjusted).

Row = (country, industry, as_of_year t). Index = daily-rebalanced equal-weight of the cell's members at t, over (D_t, D_t + h months]
with D_t = Sep 30 of t+1 (same decision date as the fundamentals panel). Outputs cumulative return, max drawdown and annualised
downside deviation for h = 12 and 24 months. Membership: for t in the panel's years (2022-2024) it is the as-of membership
(top 25 by FY t revenue known at D_t). For earlier t (2016-2021) fundamentals do not exist, so membership is back-cast from t=2022
and flagged `membership_lookahead=True`: those rows contain look-ahead in WHO is in the index and are exploratory only.
Firms need >= 80% of trading days in the window; a cell needs >= min_cell_firms priced members. Delisted firms are absent (survivorship).
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from screener.features.build_panel import decision_date
from screener.features.run_panel import ROOT, vp, load_cfg

OUT = vp("data/panel/price_targets.parquet")
CLIP = 0.5  # |daily return| capped at 50% before averaging (yfinance has split/price-print errors; max raw cell return was 9e6)


def load_prices():
    px = {}
    for f in glob.glob(str(ROOT / "data/raw/prices/*.parquet")):
        d = pd.read_parquet(f)["close"]
        d = d[~d.index.duplicated()]
        px[Path(f).stem] = d
    return px


def index_stats(rets):
    """rets: DataFrame days x firms (daily returns, NaN where no quote). Equal-weight daily rebalanced."""
    r = rets.clip(-CLIP, CLIP).mean(axis=1).dropna()  # clip bad prints / split errors
    if len(r) < 20:
        return {}
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    return {"ret": cum.iloc[-1] - 1, "mdd": dd.min(), "dd_dev": float(np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252)), "vol": float(r.std() * np.sqrt(252))}


def window(px, syms, start, end, months, fxs=None, ccy=None):
    """Daily returns (dates strictly after `start`, up to `end`) of the symbols that trade before `start` and cover >= 80% of the
    window. fxs/ccy: if given, returns are converted to USD with daily FX (units per USD, forward-filled)."""
    cols = {}
    for s in syms:
        p = px.get(s)
        if p is None or p.index.min() > start - pd.Timedelta(days=7):  # must be trading before the window starts
            continue
        w = p[(p.index >= start - pd.Timedelta(days=7)) & (p.index <= end)]
        ret = w.pct_change(fill_method=None)[w.index > start]
        if fxs is not None and ccy.get(s, "USD") != "USD":
            f = fxs[ccy[s]].reindex(w.index, method="ffill")
            ret = ((1 + ret) * (f.shift(1) / f)[w.index > start] - 1)
        if len(ret) >= 0.8 * 252 * months / 12 * 0.98:
            cols[s] = ret
    return cols


def span(cols):
    """(first, last) return date actually used - recorded for the window-leakage audit."""
    return min(c.index.min() for c in cols.values()), max(c.index.max() for c in cols.values())


def main(usd=False):
    st, _, _ = load_cfg()
    px = load_prices()
    fxs = ccy = None
    if usd:
        from screener.ingest.fx import OUT as FXDIR, listing_ccy
        fxs = {Path(f).stem: pd.read_parquet(f)["per_usd"] for f in glob.glob(str(FXDIR / "*.parquet"))}
        ccy = listing_ccy()
    obs = pd.read_parquet(vp("data/panel/firm_obs.parquet"))
    last = max(s.index.max() for s in px.values())
    mem = obs[["symbol", "country", "industry", "t"]].drop_duplicates()
    rows = []
    for (c, ind), g in mem.groupby(["country", "industry"]):
        by_t = {t: sorted(x.symbol) for t, x in g.groupby("t")}
        first = min(by_t)
        for t in range(2016, 2026):
            look = t < first
            syms = by_t.get(t) if not look else by_t[first]
            if not syms:
                continue
            D = decision_date(t, st)
            row = {"country": c, "industry": ind, "as_of_year": t, "membership_lookahead": bool(look), "decision_date": D}
            for m in (12, 24):
                end = D + pd.DateOffset(months=m)
                if end > last:
                    continue
                cols = window(px, syms, D, end, m, fxs, ccy)
                row[f"n_priced_{m}m"] = len(cols)
                if len(cols) >= st["min_cell_firms"]:
                    row.update({f"{k}_{m}m": v for k, v in index_stats(pd.DataFrame(cols)).items()})
                    row[f"fwd_first_date_{m}m"], _ = span(cols)
            # trailing 12 months ending at D (known at D): the persistence baseline for the risk targets
            tc = window(px, syms, D - pd.DateOffset(months=12), D, 12, fxs, ccy)
            if len(tc) >= st["min_cell_firms"]:
                row.update({f"trail_{k}_12m": v for k, v in index_stats(pd.DataFrame(tc)).items()})
                row["trail_first_date"], row["trail_last_date"] = span(tc)
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df[~df.country.isin(st["drop_from_panel"])]
    df.to_parquet(vp("data/panel/price_targets_usd.parquet") if usd else OUT)
    return df, px, mem


if __name__ == "__main__":
    import sys
    df, px, mem = main(usd=len(sys.argv) > 1 and sys.argv[1] == "usd")
    pd.set_option("display.width", 220)
    print("tickers with prices:", len(px), "of", mem.symbol.nunique())
    print("rows:", len(df), "| cells:", df.groupby(["country", "industry"]).ngroups)
    for tgt in ("ret_12m", "ret_24m", "mdd_12m", "dd_dev_12m"):
        v = df[df[tgt].notna()]
        print(f"{tgt}: rows {len(v)} (as-of valid {int((~v.membership_lookahead).sum())}, look-ahead membership {int(v.membership_lookahead.sum())}) "
              f"as-of years {sorted(v.as_of_year.unique())} countries {v.country.nunique()} cells {v.groupby(['country','industry']).ngroups}")
    print(df.groupby("as_of_year")[["ret_12m", "ret_24m"]].agg(lambda s: s.notna().sum()).T.to_string())

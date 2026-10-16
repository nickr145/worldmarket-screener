"""Daily FX (units of currency per USD) for the listing currencies of panel firms -> data/raw/fx/<CCY>.parquet. Used only for the
USD-converted sensitivity of the risk series."""
from pathlib import Path

import pandas as pd
import yfinance as yf

from screener.features.run_panel import vp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "fx"
NORM = {"GBp": "GBP", "ILA": "ILS", "ZAc": "ZAR"}


def listing_ccy():
    fu = pd.read_csv(ROOT / "reports/firm_universe.csv")
    return fu.set_index("symbol").ccy.map(lambda c: NORM.get(c, c))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    obs = pd.read_parquet(vp("data/panel/firm_obs.parquet"))
    ccys = sorted(set(listing_ccy().reindex(obs.symbol.unique()).dropna()) - {"USD"})
    for c in ccys:
        f = OUT / f"{c}.parquet"
        if f.exists():
            continue
        h = yf.Ticker(f"{c}=X").history(start="2014-12-01", timeout=20)["Close"]
        if len(h) < 500:
            print("FX FAIL", c, len(h))
            continue
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        h[~h.index.duplicated()].rename("per_usd").to_frame().to_parquet(f)
    print(ccys)


if __name__ == "__main__":
    main()

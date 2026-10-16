"""Write data/panel/comparable_firms.csv: firm name, ticker, country, industry for each cell and as-of year. NAMES ONLY (no financials,
prices or market caps), so the dashboard can list a cell's listed firms without any firm-level data files.
python -m screener.score.comparable_firms  (needs data/panel/firm_obs.parquet and reports/firm_universe.csv, i.e. after a panel build)"""
import pandas as pd

from screener.features.run_panel import ROOT, vp


def main():
    obs = pd.read_parquet(vp("data/panel/firm_obs.parquet"))[["symbol", "country", "industry", "t"]]
    names = pd.read_csv(ROOT / "reports/firm_universe.csv").drop_duplicates("symbol").set_index("symbol").name
    out = obs.assign(name=obs.symbol.map(names)).rename(columns={"symbol": "ticker", "t": "as_of_year"})
    out = out[["country", "industry", "as_of_year", "name", "ticker"]].sort_values(["country", "industry", "as_of_year", "name"])
    out.to_csv(vp("data/panel/comparable_firms.csv"), index=False)
    print(len(out), "rows,", out.groupby(["country", "industry"]).ngroups, "cells")


if __name__ == "__main__":
    main()

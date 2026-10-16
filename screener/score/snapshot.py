"""Write reports/data_snapshot.json: when the raw data behind scored_cells.csv was pulled (file dates of the local caches) and the last
price date. The UI shows this from the committed JSON, so a fresh clone (no raw caches) still displays it.
python -m screener.score.snapshot  (run after any refresh, before committing reports/)"""
import datetime as dt
import json

import pandas as pd

from screener.features.run_panel import ROOT, vp

RAW = ROOT / "data/raw"


def main():
    pulls = {}
    for sub in ("financials", "prices", "fx", "macro"):
        t = [f.stat().st_mtime for f in (RAW / sub).glob("*.parquet")]
        pulls[sub] = {"first": dt.date.fromtimestamp(min(t)).isoformat(), "last": dt.date.fromtimestamp(max(t)).isoformat(), "files": len(t)}
    last_price = max(pd.read_parquet(f).index.max() for f in (RAW / "prices").glob("*.parquet"))
    out = {"snapshot_date": max(v["last"] for v in pulls.values()), "latest_price_date": pd.Timestamp(last_price).date().isoformat(), "pulls": pulls,
           "note": "Dates are local cache file dates. yfinance revisions after this date are not reflected."}
    (vp("reports/data_snapshot.json")).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""Narrator evaluation. dev = the 24 cells (random_state=1) used while tuning the prompt; holdout = 40 other scored cells, run ONCE.
python -m screener.parse.holdout dev|holdout"""
import sys

import pandas as pd

from screener.features.run_panel import vp
from screener.parse.narrate import narrate
from screener.parse.parser import get_client

TUNING_EXTRA = {("SA", "software"), ("AU", "software"), ("ES", "software"), ("ID", "software"), ("JP", "software"), ("MY", "software"), ("KR", "retail")}


def cells():
    sc = pd.read_csv(vp("reports/scored_cells.csv"))
    sc = sc[~sc.insufficient_data]
    dev = sc.sample(24, random_state=1)
    devkeys = set(zip(dev.country, dev.industry)) | TUNING_EXTRA
    rest = sc[[k not in devkeys for k in zip(sc.country, sc.industry)]]
    hold = pd.concat([g.sample(10, random_state=7) for _, g in rest.groupby("industry")])  # 10 per industry
    return dev, hold


def run(which):
    dev, hold = cells()
    df = dev if which == "dev" else hold
    c = get_client()
    rows = []
    for _, r in df.iterrows():
        n = narrate(r, c)
        rows.append({"country": r.country, "industry": r.industry, "verdict": r.verdict, "source": n.source, "violations": "; ".join(n.violations),
                     "text": n.text})
    out = pd.DataFrame(rows)
    out.to_csv(vp(f"reports/narrator_{which}.csv"), index=False)  # new file name per version: the historical v1 results are never overwritten
    print(f"{which}: n={len(out)} | llm pass {int((out.source == 'llm').sum())} ({(out.source == 'llm').mean():.1%}) | template fallback "
          f"{int((out.source == 'template').sum())} ({(out.source == 'template').mean():.1%}) | verdicts {out.verdict.value_counts().to_dict()}")
    print(out[out.source == "template"][["country", "industry", "violations"]].to_string(index=False))
    return out


if __name__ == "__main__":
    run(sys.argv[1])

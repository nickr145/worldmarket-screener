"""v1.1 code must not read or write v1 files: every generated panel / price-target / firm_obs / report path goes through vp()."""
import re
from pathlib import Path

from screener.features.run_panel import TAG, vp

ROOT = Path(__file__).resolve().parents[2]
GEN = re.compile(r"(data/panel|reports|models)/")
IO = re.compile(r"read_parquet|read_csv|read_json|json\.load|to_parquet|to_csv|\bopen\(")
UNTAGGED = ("firm_universe", "fetch_failures", "currency_overrides", "parser_canned")  # raw-fetch inputs, the reviewed override table, parser-only output
SKIP = ("tests/", "compare_versions.py", "freeze.py")  # compare_versions reads v1 and v1.1 on purpose; freeze is checked below


def test_generated_paths_use_vp():
    bad = []
    for f in (ROOT / "screener").rglob("*.py"):
        rel = f.relative_to(ROOT).as_posix()
        if any(s in rel for s in SKIP):
            continue
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if IO.search(line) and GEN.search(line) and "vp(" not in line and not any(u in line for u in UNTAGGED):
                bad.append(f"{rel}:{n}: {line.strip()}")
    assert not bad, "unversioned generated path:\n" + "\n".join(bad)


def test_freeze_inputs_are_versioned():
    src = (ROOT / "screener/score/freeze.py").read_text()
    for name in ("panel", "price_targets"):
        assert f'vp("data/panel/{name}.parquet")' in src


def test_vp_tags_generated_names():
    assert TAG, "run with the default WM_TAG (v1.1)"
    assert vp("data/panel/panel.parquet").name == f"panel{TAG}.parquet"
    assert vp("data/panel/firm_obs_biotech.parquet").name == f"firm_obs_biotech{TAG}.parquet"
    assert vp("data/panel/price_targets_usd.parquet").name == f"price_targets_usd{TAG}.parquet"

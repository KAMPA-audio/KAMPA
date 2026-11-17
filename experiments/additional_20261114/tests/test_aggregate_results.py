"""Aggregation tests for notebooks/KAMPA_결과집계.py; pandas only, no torch.

Run from the package directory with:
    python -m pytest -q tests/test_aggregate_results.py

metrics.json stores the test-clip count as test['n']. The old aggregate let it
overwrite recipe['n'], so N=200 and N=2000 merged and the 2x2 step raised
TypeError. Artifact hashing is covered by test_resume_integrity.py, so
validate_complete is replaced here.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kampa_aggregate", BASE / "notebooks" / "KAMPA_결과집계.py")
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)
core = agg.load_core(BASE / "src")

FACTORIAL = ["fixed_single", "fixed_set", "meta_single", "kampa"]


def write_run(out, n, seed, variant, f1, name=None):
    recipe = dict(stage="1_3", n=n, seed=seed, variant=variant, split_seed=0)
    d = Path(out) / "runs" / (name or f"split0_n{n}_s{seed}_{variant}")
    d.mkdir(parents=True)
    test = dict(accuracy=f1, macro_precision=f1, macro_recall=f1, macro_f1=f1, confusion=[[0]], n=430)
    (d / "metrics.json").write_text(json.dumps(dict(recipe=recipe, best_epoch=3, test=test)))
    (d / "protocol.json").write_text(json.dumps(dict(runtime=dict(device="cpu"))))
    (d / "complete.json").write_text("{}")


@pytest.fixture(autouse=True)
def skip_artifact_hashes(monkeypatch):
    monkeypatch.setattr(core, "validate_complete", lambda folder, identity=None: None)


def test_label_budget_is_kept_and_effects_are_per_budget(tmp_path):
    for n, base in [(200, 0.80), (2000, 0.90)]:
        for i, variant in enumerate(FACTORIAL):
            write_run(tmp_path, n, 1, variant, base + i / 100)
    agg.aggregate(tmp_path, core)
    runs = pd.read_csv(tmp_path / "results_all_runs.csv")
    assert sorted(runs.n.unique()) == [200, 2000] and set(runs.n_test) == {430}
    summary = pd.read_csv(tmp_path / "results_summary.csv")
    assert len(summary) == 8 and set(summary.macro_f1_count) == {1}
    assert {"macro_precision_mean", "macro_recall_mean"} <= set(summary.columns)
    effects = pd.read_csv(tmp_path / "ablation_factorial_effects.csv").set_index("n")
    # meta_single - fixed_single = 0.02, kampa - fixed_set = 0.02 in both budgets.
    assert effects.loc[200, "macro_f1_meta_effect_single"] == pytest.approx(0.02)
    assert effects.loc[2000, "macro_f1_meta_effect_set"] == pytest.approx(0.02)


def test_folder_that_disagrees_with_recipe_stops(tmp_path):
    write_run(tmp_path, 200, 1, "kampa", 0.9, name="split0_n2000_s1_kampa")
    with pytest.raises(RuntimeError, match="does not match"):
        agg.aggregate(tmp_path, core)


def test_launch_patch_reaches_run_suite(monkeypatch):
    monkeypatch.setattr(core, "aggregate", lambda out: "patched")
    assert core.run_suite.__globals__["aggregate"]("x") == "patched"

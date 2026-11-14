"""Unit tests for scripts/prepare_subsets.py (CLI, overwrite guard, notebook consistency)."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

from adaptive_rag import config as cfg
from adaptive_rag.data import prepare_data

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "prepare_subsets.py"


@pytest.fixture
def script(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("prepare_subsets_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = {split: tmp_path / f"{split}_subset.parquet" for split in ("train", "val", "test")}
    monkeypatch.setattr(module, "SUBSET_PATHS", paths)
    calls = []
    monkeypatch.setattr(prepare_data, "prepare_subsets",
                        lambda configs, seed: calls.append((configs, seed)))
    module.calls, module.tmp_paths = calls, paths
    return module


def test_no_flags_builds_all_splits_at_default_sizes(script):
    assert script.main([]) == 0
    (configs, seed), = script.calls
    assert configs == {split: {"n": n, "strategy": "balanced"}
                       for split, n in script.DEFAULT_SIZES.items()}
    assert set(configs) == {"train", "val", "test"}
    assert seed == cfg.SEED


def test_size_flags_build_only_the_named_splits(script):
    assert script.main(["--val", "50"]) == 0
    (configs, _), = script.calls
    assert configs == {"val": {"n": 50, "strategy": "balanced"}}


def test_strategy_and_seed_are_forwarded(script):
    script.main(["--test", "10", "--strategy", "stratified", "--seed", "7"])
    (configs, seed), = script.calls
    assert configs["test"] == {"n": 10, "strategy": "stratified"} and seed == 7


def test_refuses_to_overwrite_existing_files(script, capsys):
    script.tmp_paths["val"].write_text("existing")
    assert script.main(["--val", "50"]) == 1
    assert script.calls == []
    assert script.tmp_paths["val"].read_text() == "existing"
    assert str(script.tmp_paths["val"]) in capsys.readouterr().err


def test_guard_only_checks_requested_splits(script):
    script.tmp_paths["train"].write_text("existing")
    assert script.main(["--val", "50"]) == 0
    assert len(script.calls) == 1


def test_default_run_is_refused_if_any_split_exists(script):
    script.tmp_paths["test"].write_text("existing")
    assert script.main([]) == 1 and script.calls == []


def test_force_allows_overwriting(script):
    script.tmp_paths["val"].write_text("existing")
    assert script.main(["--val", "50", "--force"]) == 0
    assert len(script.calls) == 1


@pytest.mark.parametrize("argv", [["--val", "0"], ["--val", "-5"], ["--train", "abc"],
                                  ["--strategy", "random"]])
def test_invalid_arguments_exit_before_doing_any_work(script, argv):
    with pytest.raises(SystemExit) as exc:
        script.main(argv)
    assert exc.value.code == 2 and script.calls == []


def test_notebook_01_has_no_hardcoded_subset_sizes():
    """Sizes have a single source of truth: the script flags and the Parquet files it writes."""
    notebook = json.loads((REPO / "notebooks" / "01_prepare_training_data.ipynb")
                          .read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                       if cell["cell_type"] == "code")
    assert not re.search(r"^(TRAIN|VAL|TEST)_SIZE\s*=", source, re.MULTILINE)
    assert "prepare_subsets(" not in source

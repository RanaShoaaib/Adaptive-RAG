"""Shared pytest setup for the Adaptive-RAG unit tests.

* Makes ``adaptive_rag`` importable from a plain checkout (no install needed).
* Replaces *heavy optional* third-party libraries (torch, transformers,
  sentence-transformers, weaviate, datasets, dotenv, tqdm) with light stubs
  **only when the real library cannot be imported**. On a full environment the
  real libraries are used; on a bare CI box the pure-Python logic is still tested.

None of these tests download models, touch the network, or need a GPU.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent
for _path in (ROOT_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _usable(name: str, *attributes: str) -> bool:
    """True if ``name`` imports for real and exposes the attributes we rely on.

    Checking attributes matters: the repo's own ``datasets/`` data folder can
    masquerade as an (empty) namespace package named ``datasets``.
    """
    try:
        module = importlib.import_module(name)
    except Exception:  # noqa: BLE001 - any import failure means "not usable"
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            sys.modules.pop(key, None)
        return False
    return all(hasattr(module, attribute) for attribute in attributes)


def _install_stub(name: str, module: types.ModuleType | None = None) -> None:
    """Register ``name`` (and its parent packages) as stub modules."""
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        sys.modules.pop(key, None)
    parts = name.split(".")
    for depth in range(1, len(parts) + 1):
        dotted = ".".join(parts[:depth])
        if dotted in sys.modules:
            continue
        stub = module if (module is not None and depth == len(parts)) else MagicMock(name=dotted)
        if isinstance(stub, MagicMock):
            stub.__path__ = []  # behave like a package so submodule imports resolve
        sys.modules[dotted] = stub


def _passthrough_tqdm() -> types.ModuleType:
    module = types.ModuleType("tqdm")
    module.tqdm = lambda iterable=None, *args, **kwargs: iterable  # type: ignore[attr-defined]
    return module


_HEAVY = {
    "torch": (("inference_mode",), None),
    "transformers": (("AutoTokenizer", "AutoModelForCausalLM"), None),
    "sentence_transformers": (("SentenceTransformer", "CrossEncoder"), None),
    "datasets": (("Dataset", "load_dataset", "concatenate_datasets"), None),
    "dotenv": (("load_dotenv",), None),
    "tqdm": (("tqdm",), _passthrough_tqdm),
}

STUBBED: list[str] = []

for _name, (_attrs, _factory) in _HEAVY.items():
    if not _usable(_name, *_attrs):
        _install_stub(_name, _factory() if _factory else None)
        STUBBED.append(_name)

if not _usable("weaviate", "connect_to_custom") or not _usable("weaviate.classes.init", "Auth") \
        or not _usable("weaviate.classes.config", "Configure"):
    for _key in [k for k in sys.modules if k == "weaviate" or k.startswith("weaviate.")]:
        sys.modules.pop(_key, None)
    for _dotted in ("weaviate", "weaviate.classes", "weaviate.classes.init", "weaviate.classes.config"):
        _install_stub(_dotted)
    STUBBED.append("weaviate")


def pytest_report_header(config):  # pragma: no cover - cosmetic
    return f"adaptive-rag tests: stubbed optional deps -> {STUBBED or 'none (real libraries in use)'}"


@pytest.fixture
def stubbed_dependencies() -> list[str]:
    """Names of third-party libraries that are stubbed in this environment."""
    return list(STUBBED)

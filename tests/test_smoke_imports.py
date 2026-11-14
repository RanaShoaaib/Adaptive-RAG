"""Import every module in the package: catches syntax errors and circular imports."""
import importlib
import pkgutil

import pytest

import adaptive_rag

MODULES = sorted(
    info.name
    for info in pkgutil.walk_packages(adaptive_rag.__path__, prefix="adaptive_rag.")
)


def test_package_modules_were_discovered():
    assert "adaptive_rag.planner.adaptive" in MODULES
    assert "adaptive_rag.evaluation.runner" in MODULES
    assert len(MODULES) >= 15


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    assert importlib.import_module(module_name) is not None

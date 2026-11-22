import importlib
import pytest

@pytest.mark.parametrize(
    "module", ["cbp", "cbp.algorithms", "cbp.model", "cbp.experiments"]
)
def test_module_is_importable(module):
    assert importlib.import_module(module) is not None
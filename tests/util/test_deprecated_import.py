from __future__ import annotations

import importlib
import sys
import warnings

import pytest

from sentence_transformers.util.deprecated_import import DEPRECATED_MODULE_PATHS


@pytest.mark.parametrize(("deprecated_path", "new_path"), DEPRECATED_MODULE_PATHS.items())
def test_deprecated_import_paths(deprecated_path: str, new_path: str):
    """Test that deprecated import paths point to the correct new paths."""
    # Clear the module from sys.modules if it exists
    if deprecated_path in sys.modules:
        del sys.modules[deprecated_path]
    if new_path in sys.modules:
        del sys.modules[new_path]

    with warnings.catch_warnings(record=True) as warnings_list:
        warnings.simplefilter("always", DeprecationWarning)

        # Import the deprecated module
        deprecated_module = importlib.import_module(deprecated_path)
        deprecation_warnings = [
            warning for warning in warnings_list if issubclass(warning.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) > 0
        assert any(
            (
                f"Importing from '{deprecated_path}' is deprecated and will be removed in a future version. "
                f"Please use '{new_path}' instead."
            )
            in str(warning.message)
            for warning in deprecation_warnings
        )
        assert deprecated_module is not None

    with warnings.catch_warnings(record=True) as warnings_list:
        warnings.simplefilter("always", DeprecationWarning)
        # Import the new module
        new_module = importlib.import_module(new_path)
        deprecation_warnings = [
            warning for warning in warnings_list if issubclass(warning.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0
        assert new_module is not None

    # Verify that both imports point to the same module
    assert deprecated_module is new_module

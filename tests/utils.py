from __future__ import annotations

import os


def is_ci() -> bool:
    """
    Check if the code is running in a Continuous Integration (CI) environment.
    This is determined by checking for the presence of certain environment variables.
    """
    return "GITHUB_ACTIONS" in os.environ

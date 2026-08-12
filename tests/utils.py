from __future__ import annotations

import os
import sys

import pytest
import torch
from packaging.version import Version

# torch 2.13 hard-crashes the process (0xc000001d) on the first bfloat16 CPU matmul on part of the
# Windows runner fleet. Gated on the version so the tests come back on their own once torch fixes
# it, rather than staying skipped forever for a transient bug.
skip_bfloat16_cpu_crash = pytest.mark.skipif(
    sys.platform == "win32" and not torch.cuda.is_available() and Version(torch.__version__) >= Version("2.13"),
    reason="bfloat16 CPU matmul hard-crashes (0xc000001d) on some Windows machines with torch>=2.13.",
)

# A chat template in the style of instruct backbones: branches on the chat roles only, so the
# query/document roles that pairs are mapped to render to nothing.
GENERIC_INSTRUCT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n{% endif %}"
    "{% endfor %}"
)
# A chat template in the style of published rerankers: selects the query/document roles explicitly.
RERANKER_TEMPLATE = (
    '<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}\n'
    '<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}'
)


def is_ci() -> bool:
    """
    Check if the code is running in a Continuous Integration (CI) environment.
    This is determined by checking for the presence of certain environment variables.
    """
    return "GITHUB_ACTIONS" in os.environ

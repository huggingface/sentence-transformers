from __future__ import annotations

import inspect
import runpy
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("sphinx")

import sentence_transformers
from sentence_transformers.util.quantization import quantize_embeddings

CONF = Path(__file__).resolve().parents[2] / "docs" / "conf.py"


def test_source_links_use_build_commit(monkeypatch, tmp_path):
    commit = "1234567890abcdef" * 2 + "12345678"
    calls = []

    def check_output(command, **kwargs):
        calls.append((command, kwargs["cwd"]))
        return commit + "\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)
    # Sphinx can be invoked from outside docs/.
    monkeypatch.chdir(tmp_path)
    config = runpy.run_path(str(CONF))
    url = config["linkcode_resolve"](
        "py", {"module": quantize_embeddings.__module__, "fullname": "quantize_embeddings"}
    )
    lines, start = inspect.getsourcelines(quantize_embeddings)
    assert url == (
        f"https://github.com/huggingface/sentence-transformers/blob/{commit}/"
        f"sentence_transformers/util/quantization.py#L{start}-L{start + len(lines) - 1}"
    )
    assert config["html_context"]["github_version"] == commit + "/"
    assert calls == [(["git", "rev-parse", "HEAD"], str(CONF.parent))]


@pytest.mark.parametrize("version, ref", [("6.2.0", "v6.2.0"), ("6.3.0.dev0", "main")])
@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.CalledProcessError(128, "git")])
def test_source_archive_fallback(monkeypatch, version, ref, error):
    def check_output(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(sentence_transformers, "__version__", version)
    config = runpy.run_path(str(CONF))
    assert config["source_ref"] == ref

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_package_import_survives_torchcodec_runtime_error() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import importlib.machinery
        import importlib.metadata
        import sys
        import types

        package_version = importlib.metadata.version

        def fake_package_version(name):
            if name == "torchcodec":
                return "0.9.1"
            return package_version(name)

        importlib.metadata.version = fake_package_version

        torchcodec = types.ModuleType("torchcodec")
        torchcodec.__path__ = []
        torchcodec.__spec__ = importlib.machinery.ModuleSpec("torchcodec", loader=None, is_package=True)

        decoders = types.ModuleType("torchcodec.decoders")
        decoders.__spec__ = importlib.machinery.ModuleSpec("torchcodec.decoders", loader=None)

        def fail_to_load_decoder(name):
            if name in {"AudioDecoder", "VideoDecoder"}:
                raise RuntimeError("Could not load libtorchcodec")
            raise AttributeError(name)

        decoders.__getattr__ = fail_to_load_decoder
        torchcodec.decoders = decoders
        sys.modules["torchcodec"] = torchcodec
        sys.modules["torchcodec.decoders"] = decoders

        from sentence_transformers import SentenceTransformer
        from transformers.audio_utils import load_audio_torchcodec

        assert SentenceTransformer is not None
        try:
            load_audio_torchcodec("broken.wav")
        except RuntimeError as error:
            assert "Could not load libtorchcodec" in str(error)
        else:
            raise AssertionError("decoder-backed audio use must preserve the torchcodec RuntimeError")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, (str(repository_root), os.environ.get("PYTHONPATH")))),
        },
        text=True,
    )

    assert result.returncode == 0, result.stderr

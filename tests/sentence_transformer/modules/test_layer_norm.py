from __future__ import annotations

import json
from pathlib import Path

import torch

from sentence_transformers.sentence_transformer.modules import LayerNorm


def test_layer_norm_load_ignores_unknown_config_keys(tmp_path: Path) -> None:
    """A config.json may record keys that ``LayerNorm.__init__`` does not accept (e.g. written by
    another tool or a future version). ``LayerNorm.load`` should drop such keys rather than crash
    in ``cls(**config)`` with a TypeError."""
    layer_norm = LayerNorm(dimension=16)
    model_dir = tmp_path / "layer_norm_extra_key"
    model_dir.mkdir()
    layer_norm.save(str(model_dir))

    # Inject a config key that ``LayerNorm.__init__`` does not accept.
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["hidden_size"] = 16
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = LayerNorm.load(str(model_dir), local_files_only=True)

    assert isinstance(loaded, LayerNorm)
    assert loaded.dimension == 16

    # The reloaded module still runs a forward pass.
    output = loaded.forward({"sentence_embedding": torch.randn(2, 16)})
    assert output["sentence_embedding"].shape == (2, 16)

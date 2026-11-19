from __future__ import annotations

import pytest


def test_pooling_name_suffix() -> None:
    from sentence_transformers.multi_vector_encoder.evaluation.utils import pooling_name_suffix
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling, LambdaTokenPooling

    assert pooling_name_suffix(HierarchicalTokenPooling(pool_factor=2)) == "pool_factor_2"
    assert pooling_name_suffix(HierarchicalTokenPooling(pool_factor=1.5)) == "pool_factor_1.5"
    assert pooling_name_suffix(HierarchicalTokenPooling(max_tokens=64)) == "max_tokens_64"
    assert pooling_name_suffix(HierarchicalTokenPooling(distance_threshold=0.25)) == "distance_threshold_0.25"
    assert pooling_name_suffix(LambdaTokenPooling(pool_fn=lambda e: e)) == "lambda_token_pooling"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tasks": ["query", "document"]},
        {"tasks": []},
        {"num_protected_tokens": 0},
        {"normalize_embeddings": False},
    ],
)
def test_pooling_names_stay_short_with_non_default_settings(kwargs) -> None:
    from sentence_transformers.multi_vector_encoder.evaluation.utils import pooling_name_suffix
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

    pooling = HierarchicalTokenPooling(max_tokens=16, **kwargs)
    assert pooling_name_suffix(pooling) == "max_tokens_16"


def test_custom_pooling_configuration() -> None:
    import json

    from sentence_transformers.multi_vector_encoder.evaluation.utils import pooling_config, pooling_name_suffix
    from sentence_transformers.multi_vector_encoder.modules import BaseTokenPooling

    class StridePooling(BaseTokenPooling):
        config_keys = ["tasks", "stride"]

        def __init__(self, stride: int) -> None:
            super().__init__()
            self.stride = stride

        def pool_one(self, embedding):
            return embedding[:: self.stride]

    first, second = StridePooling(2), StridePooling(3)
    assert pooling_name_suffix(first) == pooling_name_suffix(second) == "stride_pooling"
    config = pooling_config(first)
    assert config["type"].endswith(".StridePooling")
    assert config["config"] == {"tasks": ["document"], "stride": 2}
    assert json.loads(json.dumps(config)) == config

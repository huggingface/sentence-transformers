from __future__ import annotations

from .decorators import save_to_hub_args_decorator
from .distributed import all_gather, all_gather_with_grad
from .environment import (
    check_package_availability,
    get_device_name,
    is_accelerate_available,
    is_datasets_available,
    is_training_available,
)
from .file_io import disabled_tqdm, http_get, is_sentence_transformer_model, load_dir_path, load_file_path
from .hard_negatives import mine_hard_negatives
from .misc import append_to_last_row, disable_datasets_caching, disable_logging, fullname, import_from_string
from .retrieval import (
    community_detection,
    information_retrieval,
    paraphrase_mining,
    paraphrase_mining_embeddings,
    semantic_search,
)
from .similarity import (
    cos_sim,
    dot_score,
    euclidean_sim,
    manhattan_sim,
    pairwise_angle_sim,
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_manhattan_sim,
    pytorch_cos_sim,
)
from .tensor import (
    _convert_to_batch,
    _convert_to_batch_tensor,
    _convert_to_tensor,
    batch_to_device,
    compute_count_vector,
    normalize_embeddings,
    pad_and_stack,
    select_max_active_dims,
    to_scipy_coo,
    truncate_embeddings,
)

__all__ = [
    # From decorators.py
    "save_to_hub_args_decorator",
    # From distributed.py
    "all_gather",
    "all_gather_with_grad",
    # From environment.py
    "get_device_name",
    "check_package_availability",
    "is_accelerate_available",
    "is_datasets_available",
    "is_training_available",
    # From file_io.py
    "is_sentence_transformer_model",
    "load_dir_path",
    "load_file_path",
    "http_get",
    "disabled_tqdm",
    # From misc.py
    "fullname",
    "import_from_string",
    "disable_datasets_caching",
    "disable_logging",
    "append_to_last_row",
    # From retrieval.py
    "community_detection",
    "information_retrieval",
    "paraphrase_mining",
    "paraphrase_mining_embeddings",
    "semantic_search",
    # From similarity.py
    "cos_sim",
    "dot_score",
    "euclidean_sim",
    "manhattan_sim",
    "pairwise_angle_sim",
    "pairwise_cos_sim",
    "pairwise_dot_score",
    "pairwise_euclidean_sim",
    "pairwise_manhattan_sim",
    "pytorch_cos_sim",
    # From tensor.py
    "_convert_to_batch",
    "_convert_to_batch_tensor",
    "_convert_to_tensor",
    "batch_to_device",
    "normalize_embeddings",
    "pad_and_stack",
    "select_max_active_dims",
    "to_scipy_coo",
    "truncate_embeddings",
    "compute_count_vector",
    # From hard_negatives.py
    "mine_hard_negatives",
]

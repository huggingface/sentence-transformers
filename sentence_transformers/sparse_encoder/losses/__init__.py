from __future__ import annotations

from .csr import CSRLoss, CSRReconstructionLoss
from .flops import FlopsLoss
from .sparse_angle import SparseAnglELoss
from .sparse_cosent import SparseCoSENTLoss
from .sparse_cosine_similarity import SparseCosineSimilarityLoss
from .sparse_distill_kl_div import SparseDistillKLDivLoss
from .sparse_margin_mse import SparseMarginMSELoss
from .sparse_mse import SparseMSELoss
from .sparse_multiple_negatives_ranking import SparseMultipleNegativesRankingLoss
from .sparse_triplet import SparseTripletLoss
from .splade import SpladeLoss

from .cached_splade import CachedSpladeLoss  # isort: skip  # Avoid circular import with SpladeLoss -> FlopsLoss

__all__ = [
    "CachedSpladeLoss",
    "CSRLoss",
    "CSRReconstructionLoss",
    "SparseMultipleNegativesRankingLoss",
    "SparseCoSENTLoss",
    "SparseTripletLoss",
    "SparseMarginMSELoss",
    "SparseCosineSimilarityLoss",
    "SparseMSELoss",
    "SparseAnglELoss",
    "SparseDistillKLDivLoss",
    "FlopsLoss",
    "SpladeLoss",
]

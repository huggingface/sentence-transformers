from __future__ import annotations

from .CSRLoss import CSRLoss, CSRReconstructionLoss
from .FlopsLoss import FlopsLoss
from .SparseAnglELoss import SparseAnglELoss
from .SparseCoSENTLoss import SparseCoSENTLoss
from .SparseCosineSimilarityLoss import SparseCosineSimilarityLoss
from .SparseDistillKLDivLoss import SparseDistillKLDivLoss
from .SparseMarginMSELoss import SparseMarginMSELoss
from .SparseMSELoss import SparseMSELoss
from .SparseMultipleNegativesRankingLoss import SparseMultipleNegativesRankingLoss
from .SparseTripletLoss import SparseTripletLoss
from .SpladeLoss import SpladeLoss
from .CachedSpladeLoss import CachedSpladeLoss  # Must be after SpladeLoss to avoid circular import

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

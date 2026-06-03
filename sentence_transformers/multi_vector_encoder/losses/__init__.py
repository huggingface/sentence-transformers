from __future__ import annotations

from .cached_multiple_negatives_ranking import CachedMultiVectorMultipleNegativesRankingLoss
from .distill_kl_div import MultiVectorDistillKLDivLoss
from .margin_mse import MultiVectorMarginMSELoss
from .multiple_negatives_ranking import MultiVectorMultipleNegativesRankingLoss

__all__ = [
    "MultiVectorMultipleNegativesRankingLoss",
    "CachedMultiVectorMultipleNegativesRankingLoss",
    "MultiVectorDistillKLDivLoss",
    "MultiVectorMarginMSELoss",
]

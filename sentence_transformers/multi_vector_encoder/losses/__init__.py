from __future__ import annotations

from .cached_multiple_negatives_ranking import CachedMultiVectorMultipleNegativesRankingLoss
from .distill_kl_div import MultiVectorDistillKLDivLoss
from .margin_mse import MultiVectorMarginMSELoss
from .matryoshka import MultiVectorMatryoshkaLoss
from .multiple_negatives_ranking import MultiVectorMultipleNegativesRankingLoss
from .token_pooling import MultiVectorTokenPoolingLoss

__all__ = [
    "MultiVectorDistillKLDivLoss",
    "MultiVectorMarginMSELoss",
    "MultiVectorMatryoshkaLoss",
    "MultiVectorMultipleNegativesRankingLoss",
    "CachedMultiVectorMultipleNegativesRankingLoss",
    "MultiVectorTokenPoolingLoss",
]

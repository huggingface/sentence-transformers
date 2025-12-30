# CoSENTLoss must be imported before AnglELoss
from __future__ import annotations

from .CoSENTLoss import CoSENTLoss  # isort: skip

from .AdaptiveLayerLoss import AdaptiveLayerLoss
from .AnglELoss import AnglELoss
from .BatchAllTripletLoss import BatchAllTripletLoss
from .BatchHardSoftMarginTripletLoss import BatchHardSoftMarginTripletLoss
from .BatchHardTripletLoss import (
    BatchHardTripletLoss,
    BatchHardTripletLossDistanceFunction,
)
from .BatchSemiHardTripletLoss import BatchSemiHardTripletLoss
from .CachedGISTEmbedLoss import CachedGISTEmbedLoss
from .CachedMultipleNegativesRankingLoss import CachedMultipleNegativesRankingLoss
from .CachedMultipleNegativesSymmetricRankingLoss import (
    CachedMultipleNegativesSymmetricRankingLoss,
)
from .ContrastiveLoss import ContrastiveLoss, SiameseDistanceMetric
from .ContrastiveTensionLoss import (
    ContrastiveTensionDataLoader,
    ContrastiveTensionLoss,
    ContrastiveTensionLossInBatchNegatives,
)
from .CosineSimilarityLoss import CosineSimilarityLoss
from .DenoisingAutoEncoderLoss import DenoisingAutoEncoderLoss
from .DistillKLDivLoss import DistillKLDivLoss
from .GISTEmbedLoss import GISTEmbedLoss
from .MarginMSELoss import MarginMSELoss
from .Matryoshka2dLoss import Matryoshka2dLoss
from .MatryoshkaLoss import MatryoshkaLoss
from .MegaBatchMarginLoss import MegaBatchMarginLoss
from .MSELoss import MSELoss
from .MultipleNegativesRankingLoss import MultipleNegativesRankingLoss
from .MultipleNegativesSymmetricRankingLoss import MultipleNegativesSymmetricRankingLoss
from .OnlineContrastiveLoss import OnlineContrastiveLoss
from .SoftmaxLoss import SoftmaxLoss
from .TripletLoss import TripletDistanceMetric, TripletLoss

__all__ = [
    "AdaptiveLayerLoss",
    "CosineSimilarityLoss",
    "SoftmaxLoss",
    "MultipleNegativesRankingLoss",
    "MultipleNegativesSymmetricRankingLoss",
    "TripletLoss",
    "TripletDistanceMetric",
    "MarginMSELoss",
    "MatryoshkaLoss",
    "Matryoshka2dLoss",
    "MSELoss",
    "ContrastiveLoss",
    "SiameseDistanceMetric",
    "CachedGISTEmbedLoss",
    "CachedMultipleNegativesRankingLoss",
    "CachedMultipleNegativesSymmetricRankingLoss",
    "ContrastiveTensionLoss",
    "ContrastiveTensionLossInBatchNegatives",
    "ContrastiveTensionDataLoader",
    "CoSENTLoss",
    "AnglELoss",
    "OnlineContrastiveLoss",
    "MegaBatchMarginLoss",
    "DenoisingAutoEncoderLoss",
    "GISTEmbedLoss",
    "BatchHardTripletLoss",
    "BatchHardTripletLossDistanceFunction",
    "BatchHardSoftMarginTripletLoss",
    "BatchSemiHardTripletLoss",
    "BatchAllTripletLoss",
    "DistillKLDivLoss",
]

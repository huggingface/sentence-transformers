from __future__ import annotations

import importlib
import sys

DEPRECATED_MODULE_PATHS = {
    # Moved in Sentence Transformers v5.4.0
    "sentence_transformers.SentenceTransformer": "sentence_transformers.sentence_transformer.model",
    "sentence_transformers.sparse_encoder.SparseEncoder": "sentence_transformers.sparse_encoder.model",
    "sentence_transformers.cross_encoder.CrossEncoder": "sentence_transformers.cross_encoder.model",
    "sentence_transformers.quantization": "sentence_transformers.util.quantization",
    "sentence_transformers.similarity_functions": "sentence_transformers.util.similarity",
    "sentence_transformers.training_args": "sentence_transformers.sentence_transformer.training_args",
    "sentence_transformers.trainer": "sentence_transformers.sentence_transformer.trainer",
    "sentence_transformers.sampler": "sentence_transformers.base.sampler",
    "sentence_transformers.peft_mixin": "sentence_transformers.base.peft_mixin",
    "sentence_transformers.model_card": "sentence_transformers.sentence_transformer.model_card",
    "sentence_transformers.data_collator": "sentence_transformers.sentence_transformer.data_collator",
    "sentence_transformers.LoggingHandler": "sentence_transformers.util.logging",
    "sentence_transformers.datasets": "sentence_transformers.sentence_transformer.datasets",
    "sentence_transformers.datasets.DenoisingAutoEncoderDataset": "sentence_transformers.sentence_transformer.datasets.DenoisingAutoEncoderDataset",
    "sentence_transformers.datasets.NoDuplicatesDataLoader": "sentence_transformers.sentence_transformer.datasets.NoDuplicatesDataLoader",
    "sentence_transformers.datasets.ParallelSentencesDataset": "sentence_transformers.sentence_transformer.datasets.ParallelSentencesDataset",
    "sentence_transformers.datasets.SentenceLabelDataset": "sentence_transformers.sentence_transformer.datasets.SentenceLabelDataset",
    "sentence_transformers.datasets.SentencesDataset": "sentence_transformers.sentence_transformer.datasets.SentencesDataset",
    "sentence_transformers.evaluation": "sentence_transformers.sentence_transformer.evaluation",
    "sentence_transformers.evaluation.BinaryClassificationEvaluator": "sentence_transformers.sentence_transformer.evaluation.BinaryClassificationEvaluator",
    "sentence_transformers.evaluation.EmbeddingSimilarityEvaluator": "sentence_transformers.sentence_transformer.evaluation.EmbeddingSimilarityEvaluator",
    "sentence_transformers.evaluation.InformationRetrievalEvaluator": "sentence_transformers.sentence_transformer.evaluation.InformationRetrievalEvaluator",
    "sentence_transformers.evaluation.LabelAccuracyEvaluator": "sentence_transformers.sentence_transformer.evaluation.LabelAccuracyEvaluator",
    "sentence_transformers.evaluation.MSEEvaluator": "sentence_transformers.sentence_transformer.evaluation.MSEEvaluator",
    "sentence_transformers.evaluation.NanoBEIREvaluator": "sentence_transformers.sentence_transformer.evaluation.NanoBEIREvaluator",
    "sentence_transformers.evaluation.ParaphraseMiningEvaluator": "sentence_transformers.sentence_transformer.evaluation.ParaphraseMiningEvaluator",
    "sentence_transformers.evaluation.RerankingEvaluator": "sentence_transformers.sentence_transformer.evaluation.RerankingEvaluator",
    "sentence_transformers.evaluation.SequentialEvaluator": "sentence_transformers.base.evaluation.SequentialEvaluator",
    "sentence_transformers.evaluation.SimilarityFunction": "sentence_transformers.sentence_transformer.evaluation.SimilarityFunction",
    "sentence_transformers.evaluation.TranslationEvaluator": "sentence_transformers.sentence_transformer.evaluation.TranslationEvaluator",
    "sentence_transformers.evaluation.TripletEvaluator": "sentence_transformers.sentence_transformer.evaluation.TripletEvaluator",
    "sentence_transformers.losses": "sentence_transformers.sentence_transformer.losses",
    "sentence_transformers.losses.AdaptiveLayerLoss": "sentence_transformers.sentence_transformer.losses.AdaptiveLayerLoss",
    "sentence_transformers.losses.AnglELoss": "sentence_transformers.sentence_transformer.losses.AnglELoss",
    "sentence_transformers.losses.BatchAllTripletLoss": "sentence_transformers.sentence_transformer.losses.BatchAllTripletLoss",
    "sentence_transformers.losses.BatchHardSoftMarginTripletLoss": "sentence_transformers.sentence_transformer.losses.BatchHardSoftMarginTripletLoss",
    "sentence_transformers.losses.BatchHardTripletLoss": "sentence_transformers.sentence_transformer.losses.BatchHardTripletLoss",
    "sentence_transformers.losses.BatchSemiHardTripletLoss": "sentence_transformers.sentence_transformer.losses.BatchSemiHardTripletLoss",
    "sentence_transformers.losses.CachedGISTEmbedLoss": "sentence_transformers.sentence_transformer.losses.CachedGISTEmbedLoss",
    "sentence_transformers.losses.CachedMultipleNegativesRankingLoss": "sentence_transformers.sentence_transformer.losses.CachedMultipleNegativesRankingLoss",
    "sentence_transformers.losses.CachedMultipleNegativesSymmetricRankingLoss": "sentence_transformers.sentence_transformer.losses.CachedMultipleNegativesSymmetricRankingLoss",
    "sentence_transformers.losses.CoSENTLoss": "sentence_transformers.sentence_transformer.losses.CoSENTLoss",
    "sentence_transformers.losses.ContrastiveLoss": "sentence_transformers.sentence_transformer.losses.ContrastiveLoss",
    "sentence_transformers.losses.ContrastiveTensionLoss": "sentence_transformers.sentence_transformer.losses.ContrastiveTensionLoss",
    "sentence_transformers.losses.CosineSimilarityLoss": "sentence_transformers.sentence_transformer.losses.CosineSimilarityLoss",
    "sentence_transformers.losses.DenoisingAutoEncoderLoss": "sentence_transformers.sentence_transformer.losses.DenoisingAutoEncoderLoss",
    "sentence_transformers.losses.DistillKLDivLoss": "sentence_transformers.sentence_transformer.losses.DistillKLDivLoss",
    "sentence_transformers.losses.GISTEmbedLoss": "sentence_transformers.sentence_transformer.losses.GISTEmbedLoss",
    "sentence_transformers.losses.MSELoss": "sentence_transformers.sentence_transformer.losses.MSELoss",
    "sentence_transformers.losses.MarginMSELoss": "sentence_transformers.sentence_transformer.losses.MarginMSELoss",
    "sentence_transformers.losses.Matryoshka2dLoss": "sentence_transformers.sentence_transformer.losses.Matryoshka2dLoss",
    "sentence_transformers.losses.MatryoshkaLoss": "sentence_transformers.sentence_transformer.losses.MatryoshkaLoss",
    "sentence_transformers.losses.MegaBatchMarginLoss": "sentence_transformers.sentence_transformer.losses.MegaBatchMarginLoss",
    "sentence_transformers.losses.MultipleNegativesRankingLoss": "sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss",
    "sentence_transformers.losses.MultipleNegativesSymmetricRankingLoss": "sentence_transformers.sentence_transformer.losses.MultipleNegativesSymmetricRankingLoss",
    "sentence_transformers.losses.OnlineContrastiveLoss": "sentence_transformers.sentence_transformer.losses.OnlineContrastiveLoss",
    "sentence_transformers.losses.SoftmaxLoss": "sentence_transformers.sentence_transformer.losses.SoftmaxLoss",
    "sentence_transformers.losses.TripletLoss": "sentence_transformers.sentence_transformer.losses.TripletLoss",
    "sentence_transformers.models": "sentence_transformers.sentence_transformer.models",
    "sentence_transformers.models.Asym": "sentence_transformers.base.models.Router",
    "sentence_transformers.models.BoW": "sentence_transformers.sentence_transformer.models.BoW",
    "sentence_transformers.models.CLIPModel": "sentence_transformers.sentence_transformer.models.CLIPModel",
    "sentence_transformers.models.CNN": "sentence_transformers.sentence_transformer.models.CNN",
    "sentence_transformers.models.Dense": "sentence_transformers.sentence_transformer.models.Dense",
    "sentence_transformers.models.Dropout": "sentence_transformers.sentence_transformer.models.Dropout",
    "sentence_transformers.models.InputModule": "sentence_transformers.base.models.InputModule",
    "sentence_transformers.models.LSTM": "sentence_transformers.sentence_transformer.models.LSTM",
    "sentence_transformers.models.LayerNorm": "sentence_transformers.sentence_transformer.models.LayerNorm",
    "sentence_transformers.models.Module": "sentence_transformers.base.models.Module",
    "sentence_transformers.models.Normalize": "sentence_transformers.sentence_transformer.models.Normalize",
    "sentence_transformers.models.Pooling": "sentence_transformers.sentence_transformer.models.Pooling",
    "sentence_transformers.models.Router": "sentence_transformers.base.models.Router",
    "sentence_transformers.models.StaticEmbedding": "sentence_transformers.sentence_transformer.models.StaticEmbedding",
    "sentence_transformers.models.Transformer": "sentence_transformers.base.models.Transformer",
    "sentence_transformers.models.WeightedLayerPooling": "sentence_transformers.sentence_transformer.models.WeightedLayerPooling",
    "sentence_transformers.models.WordEmbeddings": "sentence_transformers.sentence_transformer.models.WordEmbeddings",
    "sentence_transformers.models.WordWeights": "sentence_transformers.sentence_transformer.models.WordWeights",
    "sentence_transformers.models.tokenizer": "sentence_transformers.sentence_transformer.models.tokenizer",
    "sentence_transformers.models.tokenizer.PhraseTokenizer": "sentence_transformers.sentence_transformer.models.tokenizer.PhraseTokenizer",
    "sentence_transformers.models.tokenizer.WhitespaceTokenizer": "sentence_transformers.sentence_transformer.models.tokenizer.WhitespaceTokenizer",
    "sentence_transformers.models.tokenizer.WordTokenizer": "sentence_transformers.sentence_transformer.models.tokenizer.WordTokenizer",
    "sentence_transformers.readers": "sentence_transformers.sentence_transformer.readers",
    "sentence_transformers.readers.InputExample": "sentence_transformers.sentence_transformer.readers.InputExample",
    "sentence_transformers.readers.LabelSentenceReader": "sentence_transformers.sentence_transformer.readers.LabelSentenceReader",
    "sentence_transformers.readers.PairedFilesReader": "sentence_transformers.sentence_transformer.readers.PairedFilesReader",
    "sentence_transformers.readers.NLIDataReader": "sentence_transformers.sentence_transformer.readers.NLIDataReader",
    "sentence_transformers.readers.STSDataReader": "sentence_transformers.sentence_transformer.readers.STSDataReader",
    "sentence_transformers.readers.TripletReader": "sentence_transformers.sentence_transformer.readers.TripletReader",
    # Deprecated in Sentence Transformers v4.0.0
    "sentence_transformers.cross_encoder.evaluation.CEBinaryAccuracyEvaluator": "sentence_transformers.cross_encoder.evaluation.deprecated",
    "sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator": "sentence_transformers.cross_encoder.evaluation.deprecated",
    "sentence_transformers.cross_encoder.evaluation.CEF1Evaluator": "sentence_transformers.cross_encoder.evaluation.deprecated",
    "sentence_transformers.cross_encoder.evaluation.CESoftmaxAccuracyEvaluator": "sentence_transformers.cross_encoder.evaluation.deprecated",
    "sentence_transformers.cross_encoder.evaluation.CECorrelationEvaluator": "sentence_transformers.cross_encoder.evaluation.deprecated",
    "sentence_transformers.cross_encoder.evaluation.CERerankingEvaluator": "sentence_transformers.cross_encoder.evaluation.deprecated",
    # Deprecated in Sentence Transformers TODO
    # TODO: Perhaps import Asym/Router etc. in sentence_transformers/sentence_transformer/models/__init__.py to allow
    # for importing from sentence_transformers.sentence_transformer.models etc.?
    "sentence_transformers.sentence_transformer.models.Asym": "sentence_transformers.base.models.Router",
    "sentence_transformers.sentence_transformer.models.Router": "sentence_transformers.base.models.Router",
}


def setup_deprecated_module_imports() -> None:
    """
    Set up deprecated module imports by directly aliasing sys.modules entries.
    """

    for old_path, new_path in DEPRECATED_MODULE_PATHS.items():
        # Import the new module if not already imported
        if new_path not in sys.modules:
            try:
                importlib.import_module(new_path)
            except ImportError:
                # If the new module doesn't exist, skip it
                continue

        sys.modules[old_path] = sys.modules[new_path]

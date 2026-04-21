from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import types
import warnings

# Map old to new module paths for deprecated imports
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
    "sentence_transformers.datasets.DenoisingAutoEncoderDataset": "sentence_transformers.sentence_transformer.datasets.denoising_auto_encoder",
    "sentence_transformers.datasets.NoDuplicatesDataLoader": "sentence_transformers.sentence_transformer.datasets.no_duplicates_dataloader",
    "sentence_transformers.datasets.ParallelSentencesDataset": "sentence_transformers.sentence_transformer.datasets.parallel_sentences",
    "sentence_transformers.datasets.SentenceLabelDataset": "sentence_transformers.sentence_transformer.datasets.sentence_label",
    "sentence_transformers.datasets.SentencesDataset": "sentence_transformers.sentence_transformer.datasets.sentences",
    "sentence_transformers.evaluation": "sentence_transformers.sentence_transformer.evaluation",
    "sentence_transformers.evaluation.BinaryClassificationEvaluator": "sentence_transformers.sentence_transformer.evaluation.binary_classification",
    "sentence_transformers.evaluation.EmbeddingSimilarityEvaluator": "sentence_transformers.sentence_transformer.evaluation.embedding_similarity",
    "sentence_transformers.evaluation.InformationRetrievalEvaluator": "sentence_transformers.sentence_transformer.evaluation.information_retrieval",
    "sentence_transformers.evaluation.LabelAccuracyEvaluator": "sentence_transformers.sentence_transformer.evaluation.label_accuracy",
    "sentence_transformers.evaluation.MSEEvaluator": "sentence_transformers.sentence_transformer.evaluation.mse",
    "sentence_transformers.evaluation.NanoBEIREvaluator": "sentence_transformers.sentence_transformer.evaluation.nano_beir",
    "sentence_transformers.evaluation.ParaphraseMiningEvaluator": "sentence_transformers.sentence_transformer.evaluation.paraphrase_mining",
    "sentence_transformers.evaluation.RerankingEvaluator": "sentence_transformers.sentence_transformer.evaluation.reranking",
    "sentence_transformers.evaluation.SentenceEvaluator": "sentence_transformers.base.evaluation.evaluator",
    "sentence_transformers.evaluation.SequentialEvaluator": "sentence_transformers.base.evaluation.sequential",
    "sentence_transformers.evaluation.SimilarityFunction": "sentence_transformers.util.similarity",
    "sentence_transformers.evaluation.TranslationEvaluator": "sentence_transformers.sentence_transformer.evaluation.translation",
    "sentence_transformers.evaluation.TripletEvaluator": "sentence_transformers.sentence_transformer.evaluation.triplet",
    "sentence_transformers.losses": "sentence_transformers.sentence_transformer.losses",
    "sentence_transformers.losses.AdaptiveLayerLoss": "sentence_transformers.sentence_transformer.losses.adaptive_layer",
    "sentence_transformers.losses.AnglELoss": "sentence_transformers.sentence_transformer.losses.angle",
    "sentence_transformers.losses.BatchAllTripletLoss": "sentence_transformers.sentence_transformer.losses.batch_all_triplet",
    "sentence_transformers.losses.BatchHardSoftMarginTripletLoss": "sentence_transformers.sentence_transformer.losses.batch_hard_soft_margin_triplet",
    "sentence_transformers.losses.BatchHardTripletLoss": "sentence_transformers.sentence_transformer.losses.batch_hard_triplet",
    "sentence_transformers.losses.BatchSemiHardTripletLoss": "sentence_transformers.sentence_transformer.losses.batch_semi_hard_triplet",
    "sentence_transformers.losses.CachedGISTEmbedLoss": "sentence_transformers.sentence_transformer.losses.cached_gist_embed",
    "sentence_transformers.losses.CachedMultipleNegativesRankingLoss": "sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking",
    "sentence_transformers.losses.CachedMultipleNegativesSymmetricRankingLoss": "sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_symmetric_ranking",
    "sentence_transformers.losses.CoSENTLoss": "sentence_transformers.sentence_transformer.losses.cosent",
    "sentence_transformers.losses.ContrastiveLoss": "sentence_transformers.sentence_transformer.losses.contrastive",
    "sentence_transformers.losses.ContrastiveTensionLoss": "sentence_transformers.sentence_transformer.losses.contrastive_tension",
    "sentence_transformers.losses.CosineSimilarityLoss": "sentence_transformers.sentence_transformer.losses.cosine_similarity",
    "sentence_transformers.losses.DenoisingAutoEncoderLoss": "sentence_transformers.sentence_transformer.losses.denoising_auto_encoder",
    "sentence_transformers.losses.DistillKLDivLoss": "sentence_transformers.sentence_transformer.losses.distill_kl_div",
    "sentence_transformers.losses.GISTEmbedLoss": "sentence_transformers.sentence_transformer.losses.gist_embed",
    "sentence_transformers.losses.MSELoss": "sentence_transformers.sentence_transformer.losses.mse",
    "sentence_transformers.losses.MarginMSELoss": "sentence_transformers.sentence_transformer.losses.margin_mse",
    "sentence_transformers.losses.Matryoshka2dLoss": "sentence_transformers.sentence_transformer.losses.matryoshka_2d",
    "sentence_transformers.losses.MatryoshkaLoss": "sentence_transformers.sentence_transformer.losses.matryoshka",
    "sentence_transformers.losses.MegaBatchMarginLoss": "sentence_transformers.sentence_transformer.losses.mega_batch_margin",
    "sentence_transformers.losses.MultipleNegativesRankingLoss": "sentence_transformers.sentence_transformer.losses.multiple_negatives_ranking",
    "sentence_transformers.losses.MultipleNegativesSymmetricRankingLoss": "sentence_transformers.sentence_transformer.losses.multiple_negatives_symmetric_ranking",
    "sentence_transformers.losses.OnlineContrastiveLoss": "sentence_transformers.sentence_transformer.losses.online_contrastive",
    "sentence_transformers.losses.SoftmaxLoss": "sentence_transformers.sentence_transformer.losses.softmax",
    "sentence_transformers.losses.TripletLoss": "sentence_transformers.sentence_transformer.losses.triplet",
    "sentence_transformers.models": "sentence_transformers.sentence_transformer.modules",
    "sentence_transformers.models.Asym": "sentence_transformers.base.modules.router",
    "sentence_transformers.models.BoW": "sentence_transformers.sentence_transformer.modules.bow",
    "sentence_transformers.models.CLIPModel": "sentence_transformers.sentence_transformer.modules.clip_model",
    "sentence_transformers.models.CNN": "sentence_transformers.sentence_transformer.modules.cnn",
    "sentence_transformers.models.Dense": "sentence_transformers.base.modules.dense",
    "sentence_transformers.models.Dropout": "sentence_transformers.sentence_transformer.modules.dropout",
    "sentence_transformers.models.InputModule": "sentence_transformers.base.modules.input_module",
    "sentence_transformers.models.LSTM": "sentence_transformers.sentence_transformer.modules.lstm",
    "sentence_transformers.models.LayerNorm": "sentence_transformers.sentence_transformer.modules.layer_norm",
    "sentence_transformers.models.Module": "sentence_transformers.base.modules.module",
    "sentence_transformers.models.Normalize": "sentence_transformers.sentence_transformer.modules.normalize",
    "sentence_transformers.models.Pooling": "sentence_transformers.sentence_transformer.modules.pooling",
    "sentence_transformers.models.Router": "sentence_transformers.base.modules.router",
    "sentence_transformers.models.StaticEmbedding": "sentence_transformers.sentence_transformer.modules.static_embedding",
    "sentence_transformers.models.Transformer": "sentence_transformers.base.modules.transformer",
    "sentence_transformers.models.WeightedLayerPooling": "sentence_transformers.sentence_transformer.modules.weighted_layer_pooling",
    "sentence_transformers.models.WordEmbeddings": "sentence_transformers.sentence_transformer.modules.word_embeddings",
    "sentence_transformers.models.WordWeights": "sentence_transformers.sentence_transformer.modules.word_weights",
    "sentence_transformers.models.tokenizer": "sentence_transformers.sentence_transformer.modules.tokenizer",
    "sentence_transformers.models.tokenizer.PhraseTokenizer": "sentence_transformers.sentence_transformer.modules.tokenizer.phrase",
    "sentence_transformers.models.tokenizer.WhitespaceTokenizer": "sentence_transformers.sentence_transformer.modules.tokenizer.whitespace",
    "sentence_transformers.models.tokenizer.WordTokenizer": "sentence_transformers.sentence_transformer.modules.tokenizer.word",
    "sentence_transformers.readers": "sentence_transformers.sentence_transformer.readers",
    "sentence_transformers.readers.InputExample": "sentence_transformers.sentence_transformer.readers.input_example",
    "sentence_transformers.readers.LabelSentenceReader": "sentence_transformers.sentence_transformer.readers.label_sentence",
    "sentence_transformers.readers.PairedFilesReader": "sentence_transformers.sentence_transformer.readers.paired_files",
    "sentence_transformers.readers.NLIDataReader": "sentence_transformers.sentence_transformer.readers.nli_data",
    "sentence_transformers.readers.STSDataReader": "sentence_transformers.sentence_transformer.readers.sts_data",
    "sentence_transformers.readers.TripletReader": "sentence_transformers.sentence_transformer.readers.triplet",
    "sentence_transformers.sparse_encoder.models": "sentence_transformers.sparse_encoder.modules",
    "sentence_transformers.sparse_encoder.models.MLMTransformer": "sentence_transformers.sparse_encoder.modules.mlm_transformer",
    "sentence_transformers.sparse_encoder.models.SparseAutoEncoder": "sentence_transformers.sparse_encoder.modules.sparse_auto_encoder",
    "sentence_transformers.sparse_encoder.models.SparseStaticEmbedding": "sentence_transformers.sparse_encoder.modules.sparse_static_embedding",
    "sentence_transformers.sparse_encoder.models.SpladePooling": "sentence_transformers.sparse_encoder.modules.splade_pooling",
    # Renamed in Sentence Transformers v5.4.0 (approximately TitleCase -> snake_case)
    "sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss": "sentence_transformers.cross_encoder.losses.binary_cross_entropy",
    "sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss": "sentence_transformers.cross_encoder.losses.cached_multiple_negatives_ranking",
    "sentence_transformers.cross_encoder.losses.CrossEntropyLoss": "sentence_transformers.cross_encoder.losses.cross_entropy",
    "sentence_transformers.cross_encoder.losses.LambdaLoss": "sentence_transformers.cross_encoder.losses.lambda_loss",
    "sentence_transformers.cross_encoder.losses.ListMLELoss": "sentence_transformers.cross_encoder.losses.list_mle",
    "sentence_transformers.cross_encoder.losses.ListNetLoss": "sentence_transformers.cross_encoder.losses.list_net",
    "sentence_transformers.cross_encoder.losses.MSELoss": "sentence_transformers.cross_encoder.losses.mse",
    "sentence_transformers.cross_encoder.losses.MarginMSELoss": "sentence_transformers.cross_encoder.losses.margin_mse",
    "sentence_transformers.cross_encoder.losses.MultipleNegativesRankingLoss": "sentence_transformers.cross_encoder.losses.multiple_negatives_ranking",
    "sentence_transformers.cross_encoder.losses.PListMLELoss": "sentence_transformers.cross_encoder.losses.plist_mle",
    "sentence_transformers.cross_encoder.losses.RankNetLoss": "sentence_transformers.cross_encoder.losses.rank_net",
    "sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator": "sentence_transformers.sparse_encoder.evaluation.reciprocal_rank_fusion",
    "sentence_transformers.sparse_encoder.evaluation.SparseBinaryClassificationEvaluator": "sentence_transformers.sparse_encoder.evaluation.sparse_binary_classification",
    "sentence_transformers.sparse_encoder.evaluation.SparseEmbeddingSimilarityEvaluator": "sentence_transformers.sparse_encoder.evaluation.sparse_embedding_similarity",
    "sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator": "sentence_transformers.sparse_encoder.evaluation.sparse_information_retrieval",
    "sentence_transformers.sparse_encoder.evaluation.SparseMSEEvaluator": "sentence_transformers.sparse_encoder.evaluation.sparse_mse",
    "sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator": "sentence_transformers.sparse_encoder.evaluation.sparse_nano_beir",
    "sentence_transformers.sparse_encoder.evaluation.SparseRerankingEvaluator": "sentence_transformers.sparse_encoder.evaluation.sparse_reranking",
    "sentence_transformers.sparse_encoder.evaluation.SparseTranslationEvaluator": "sentence_transformers.sparse_encoder.evaluation.sparse_translation",
    "sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator": "sentence_transformers.sparse_encoder.evaluation.sparse_triplet",
    "sentence_transformers.sparse_encoder.losses.CSRLoss": "sentence_transformers.sparse_encoder.losses.csr",
    "sentence_transformers.sparse_encoder.losses.CachedSpladeLoss": "sentence_transformers.sparse_encoder.losses.cached_splade",
    "sentence_transformers.sparse_encoder.losses.FlopsLoss": "sentence_transformers.sparse_encoder.losses.flops",
    "sentence_transformers.sparse_encoder.losses.SparseAnglELoss": "sentence_transformers.sparse_encoder.losses.sparse_angle",
    "sentence_transformers.sparse_encoder.losses.SparseCoSENTLoss": "sentence_transformers.sparse_encoder.losses.sparse_cosent",
    "sentence_transformers.sparse_encoder.losses.SparseCosineSimilarityLoss": "sentence_transformers.sparse_encoder.losses.sparse_cosine_similarity",
    "sentence_transformers.sparse_encoder.losses.SparseDistillKLDivLoss": "sentence_transformers.sparse_encoder.losses.sparse_distill_kl_div",
    "sentence_transformers.sparse_encoder.losses.SparseMSELoss": "sentence_transformers.sparse_encoder.losses.sparse_mse",
    "sentence_transformers.sparse_encoder.losses.SparseMarginMSELoss": "sentence_transformers.sparse_encoder.losses.sparse_margin_mse",
    "sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss": "sentence_transformers.sparse_encoder.losses.sparse_multiple_negatives_ranking",
    "sentence_transformers.sparse_encoder.losses.SparseTripletLoss": "sentence_transformers.sparse_encoder.losses.sparse_triplet",
    "sentence_transformers.sparse_encoder.losses.SpladeLoss": "sentence_transformers.sparse_encoder.losses.splade",
    # Deprecated in Sentence Transformers v4.0.0
    "sentence_transformers.cross_encoder.evaluation.CEBinaryAccuracyEvaluator": "sentence_transformers.cross_encoder.evaluation.deprecated",
    "sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator": "sentence_transformers.cross_encoder.evaluation.deprecated",
    "sentence_transformers.cross_encoder.evaluation.CEF1Evaluator": "sentence_transformers.cross_encoder.evaluation.deprecated",
    "sentence_transformers.cross_encoder.evaluation.CESoftmaxAccuracyEvaluator": "sentence_transformers.cross_encoder.evaluation.deprecated",
    "sentence_transformers.cross_encoder.evaluation.CECorrelationEvaluator": "sentence_transformers.cross_encoder.evaluation.deprecated",
    "sentence_transformers.cross_encoder.evaluation.CERerankingEvaluator": "sentence_transformers.cross_encoder.evaluation.deprecated",
}


_SENTINEL = object()
_PROTECTED_ATTRS: dict[int, dict[str, object]] = {}
_ORIGINAL_MODULE_CLASSES: dict[int, type] = {}


class _ProtectedModule(types.ModuleType):
    """Module subclass that prevents the import machinery from overwriting class attributes with modules.

    When Python imports a submodule ``parent.Child``, it does ``setattr(parent, 'Child', child_module)``.
    For deprecated paths where ``Child`` is both a deprecated submodule path AND a class exported by
    the parent's ``__init__.py``, this would overwrite the class. This subclass intercepts that
    ``setattr`` and restores the original class attribute, then reverts the module to its original class.
    """

    def __setattr__(self, name: str, value: object) -> None:
        protected = _PROTECTED_ATTRS.get(id(self))
        if protected is not None and name in protected:
            original = protected.pop(name)
            try:
                # Restore the original attribute before reverting the class, since super() needs
                # self to still be an instance of _ProtectedModule.
                super().__setattr__(name, original)
            finally:
                if not protected:
                    del _PROTECTED_ATTRS[id(self)]
                    self.__class__ = _ORIGINAL_MODULE_CLASSES.pop(id(self), types.ModuleType)
            return
        super().__setattr__(name, value)


class _DeprecatedModuleLoader(importlib.abc.Loader):
    """Loader that returns an already-loaded module without re-executing it.

    We save the original ``__spec__`` and restore it in ``exec_module`` as a safety measure,
    ensuring the module's spec always points to its canonical name regardless of how
    ``_init_module_attrs`` handles it.
    """

    def __init__(self, module: types.ModuleType) -> None:
        self._module = module
        self._original_spec = getattr(module, "__spec__", None)

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):
        if self._original_spec is not None:
            module.__spec__ = self._original_spec


class _DeprecatedModuleFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that intercepts imports of deprecated module paths.

    Issues a DeprecationWarning on first import and returns a spec whose loader provides the
    real module. Protects parent packages from having their class attributes overwritten by
    the import machinery's ``setattr(parent, child, module)`` call.
    After the first import, ``sys.modules`` has the alias, so the finder is never called again
    for that path. No overhead after the first import.
    """

    def find_spec(self, fullname, path=None, target=None):
        new_path = DEPRECATED_MODULE_PATHS.get(fullname)
        if new_path is None:
            return None

        # Walk the stack to find the caller outside of importlib internals and this module,
        # so the warning points to the user's code.
        frame = sys._getframe(1)
        current_file = __file__
        while frame is not None:
            filename = frame.f_code.co_filename
            if "importlib" not in filename and filename != current_file:
                break
            frame = frame.f_back

        msg = (
            f"Importing from '{fullname}' is deprecated and will be removed in a future version. "
            f"Please use '{new_path}' instead."
        )
        if frame is not None:
            warnings.warn_explicit(
                msg,
                DeprecationWarning,
                filename=frame.f_code.co_filename,
                lineno=frame.f_lineno,
                module=frame.f_globals.get("__name__") or frame.f_code.co_filename,
            )
        else:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        # Import the new module and alias the deprecated path to it
        new_module = importlib.import_module(new_path)

        # Protect parent's class/function attributes from being overwritten by the import
        # machinery's ``setattr(parent, child_name, module)`` after loading.
        parent_name, _, child_name = fullname.rpartition(".")
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is not None:
                original = getattr(parent, child_name, _SENTINEL)
                if original is not _SENTINEL and not isinstance(original, types.ModuleType):
                    module_id = id(parent)
                    if module_id not in _PROTECTED_ATTRS:
                        _PROTECTED_ATTRS[module_id] = {}
                        _ORIGINAL_MODULE_CLASSES[module_id] = type(parent)
                        parent.__class__ = _ProtectedModule
                    _PROTECTED_ATTRS[module_id][child_name] = original

        return importlib.util.spec_from_loader(fullname, _DeprecatedModuleLoader(new_module))


def setup_deprecated_module_imports() -> None:
    """Install a meta path finder that issues deprecation warnings for deprecated import paths
    and aliases them to their new locations in ``sys.modules``.
    """
    if not any(isinstance(f, _DeprecatedModuleFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _DeprecatedModuleFinder())

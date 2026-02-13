from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from torch import Tensor, nn
from transformers.utils import logging

from sentence_transformers.base.models.InputModule import InputModule
from sentence_transformers.base.models.modality_utils import MODALITY_TO_PROCESSOR_ARG, Modality  # , infer_modality
from sentence_transformers.base.models.Module import Module
from sentence_transformers.util import import_from_string, load_dir_path

logger = logging.get_logger(__name__)


class Router(InputModule):
    # TODO: Update the __repr__, it likely got overridden by (Input)Module instead of letting nn.Module handle it
    # Also update sparse_encoder/training_overview 'Inference-free Splade' section
    forward_kwargs = {"task", "modality"}
    config_keys: list[str] = ["default_route", "allow_empty_key", "route_mappings"]
    config_file_name = "router_config.json"

    def __init__(
        self,
        sub_modules: dict[str, list[Module]],
        default_route: str | None = None,
        allow_empty_key: bool = True,
        route_mappings: dict[tuple[str | None, str | tuple[str, ...] | None], str] | None = None,
    ) -> None:
        r"""
        This model allows creating flexible SentenceTransformer models that dynamically route inputs to different
        processing modules based on:

        1. Task type (e.g., "query" or "document") for asymmetric retrieval models
        2. Modality (e.g., "text", "image", or ("text", "image")) for crossmodal or multimodal models
        3. Combination of both for complex routing scenarios

        Tips:

        - The ``task`` argument in ``model.encode()`` specifies which route to use
        - ``model.encode_query()`` and ``model.encode_document()`` are convenient shorthands for ``task="query"`` and ``task="document"``
        - Modality is automatically inferred from input data (text strings, PIL Images, etc.)
        - You can override automatic inference by passing ``modality`` in ``model.encode()`` (and its variants) explicitly

        Route Priority:

        1. Exact match: ``(task, modality)`` - e.g., ``("query", "text")``
        2. Task with any modality: ``(task, None)`` - e.g., ``("query", None)``
        3. Any task with modality: ``(None, modality)`` - e.g., ``(None, "image")``
        4. Catch-all: ``(None, None)``
        5. Direct lookup by task name in ``sub_modules``
        6. Direct lookup by modality name in ``sub_modules``
        7. Fall back to ``default_route`` if set

        In the below examples, the ``Router`` model is used to create asymmetric models with different encoders for
        queries and documents. In these examples, the "query" route is efficient (e.g., using SparseStaticEmbedding),
        while the "document" route uses a more complex model (e.g. a Transformers module). This allows for efficient
        query encoding while still using a powerful document encoder, but the combinations are not limited to this.

        Example:
            ::

                from sentence_transformers import SentenceTransformer
                from sentence_transformers.base.models import Router, Normalize

                # Use a regular SentenceTransformer for the document embeddings, and a static embedding model for the query embeddings
                document_embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
                query_embedder = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")
                router = Router.for_query_document(
                    query_modules=list(query_embedder.children()),
                    document_modules=list(document_embedder.children()),
                )
                normalize = Normalize()

                # Create an asymmetric model with different encoders for queries and documents
                model = SentenceTransformer(
                    modules=[router, normalize],
                )

                # ... requires more training to align the vector spaces

                # Use the query & document routes
                query_embedding = model.encode_query("What is the capital of France?")
                document_embedding = model.encode_document("Paris is the capital of France.")

            ::

                from sentence_transformers.base.models import Router, Transformer
                from sentence_transformers.sparse_encoder import SparseEncoder
                from sentence_transformers.sparse_encoder.models import SparseStaticEmbedding, SpladePooling

                # Load an asymmetric model with different encoders for queries and documents
                doc_encoder = Transformer("opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill", transformer_task="fill-mask")
                router = Router.for_query_document(
                    query_modules=[
                        SparseStaticEmbedding.from_json(
                            "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
                            tokenizer=doc_encoder.tokenizer,
                            frozen=True,
                        ),
                    ],
                    document_modules=[
                        doc_encoder,
                        SpladePooling(pooling_strategy="max", activation_function="log1p_relu"),
                    ],
                )

                model = SparseEncoder(modules=[router], similarity_fn_name="dot")

                query = "What's the weather in ny now?"
                document = "Currently New York is rainy."

                query_embed = model.encode_query(query)
                document_embed = model.encode_document(document)

                sim = model.similarity(query_embed, document_embed)
                print(f"Similarity: {sim}")

                # Visualize top tokens for each text
                top_k = 10
                print(f"Top tokens {top_k} for each text:")

                decoded_query = model.decode(query_embed, top_k=top_k)
                decoded_document = model.decode(document_embed)

                for i in range(min(top_k, len(decoded_query))):
                    query_token, query_score = decoded_query[i]
                    doc_score = next((score for token, score in decoded_document if token == query_token), 0)
                    if doc_score != 0:
                        print(f"Token: {query_token}, Query score: {query_score:.4f}, Document score: {doc_score:.4f}")

                '''
                Similarity: tensor([[11.1105]], device='cuda:0')
                Top tokens 10 for each text:
                Token: ny, Query score: 5.7729, Document score: 0.8049
                Token: weather, Query score: 4.5684, Document score: 0.9710
                Token: now, Query score: 3.5895, Document score: 0.4720
                Token: ?, Query score: 3.3313, Document score: 0.0286
                Token: what, Query score: 2.7699, Document score: 0.0787
                Token: in, Query score: 0.4989, Document score: 0.0417
                '''

            Multimodal Example:

            ::

                from PIL import Image
                from sentence_transformers import SentenceTransformer
                from sentence_transformers.sentence_transformer.models import Dense, Pooling, Router, Transformer

                # Create separate encoders for different modalities
                text_encoder = Transformer("sentence-transformers/all-MiniLM-L6-v2")
                # Project to 768 dims to match image encoder
                text_dense = Dense(text_encoder.get_word_embedding_dimension(), 768, module_input_name="token_embeddings")
                image_encoder = Transformer(
                    "ModernVBERT/modernvbert",
                    model_args={"trust_remote_code": True},
                    tokenizer_args={"trust_remote_code": True},
                    config_args={"trust_remote_code": True},
                )
                pooling = Pooling(text_encoder.get_word_embedding_dimension())

                # Route based on modality
                router = Router(
                    sub_modules={
                        "text": [text_encoder, text_dense],
                        "image": [image_encoder],
                    },
                    route_mappings={
                        (None, "text"): "text",  # Any task with text goes to text encoder
                        (None, ("text", "image")): "image",  # Any task with text-image together goes to image encoder
                    },
                )

                model = SentenceTransformer(modules=[router, pooling])

                # Modality is automatically inferred
                text_embedding = model.encode("A photo of a cat")
                multimodal_embedding = model.encode({"text": "A photo of a <image>", "image": Image.open("cat.jpg")})

                # Compute the similarity; it'll be poor as the model hasn't yet been trained
                similarity = model.similarity(text_embedding, multimodal_embedding)

            Hybrid Asymmetric + Multimodal Example:

            ::

                from sentence_transformers import SentenceTransformer
                from sentence_transformers.base.models import Router

                # Different encoders for query text, document text, and images
                router = Router(
                    sub_modules={
                        "query_text": [query_text_modules],
                        "doc_text": [document_text_modules],
                        "image": [image_modules],
                    },
                    route_mappings={
                        ("query", "text"): "query_text",        # Query text uses efficient encoder
                        ("document", "text"): "doc_text",       # Document text uses powerful encoder
                        (None, ("text", "image")): "image",     # Any text-image together goes to image encoder
                    },
                )

                model = SentenceTransformer(modules=[router])

                # Explicit task + automatic modality inference
                query_embedding = model.encode_query("Find images of cats")
                doc_embedding = model.encode_document("Article about cats")
                multimodal_embedding = model.encode({"text": "A photo of a cat", "image": Image.open("cat.jpg")})

        .. note::

            When training models with the :class:`~sentence_transformers.base.models.Router` module, you must use the
            ``router_mapping`` argument in the :class:`~sentence_transformers.sentence_transformer.training_args.SentenceTransformerTrainingArguments`
            or :class:`~sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments` to map the
            training dataset columns to the correct route ("query" or "document"). For example, if your training dataset(s)
            have ``["question", "positive", "negative"]`` columns, then you can use the following mapping::

                args = SparseEncoderTrainingArguments(
                    ...,
                    router_mapping={
                        "question": "query",
                        "positive": "document",
                        "negative": "document",
                    }
                )

            Additionally, it is common to use a different learning rate for the different routes. For this, you should
            use the ``learning_rate_mapping`` argument in the :class:`~sentence_transformers.sentence_transformer.training_args.SentenceTransformerTrainingArguments`
            or :class:`~sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments` to map parameter patterns
            to their learning rates. For example, if you want to use a learning rate of ``1e-3`` for an SparseStaticEmbedding module and
            ``2e-5`` for the rest of the model, you can do this::

                args = SparseEncoderTrainingArguments(
                    ...,
                    learning_rate=2e-5,
                    learning_rate_mapping={
                        r"SparseStaticEmbedding\.*": 1e-3,
                    }
                )

        Args:
            sub_modules: Mapping of route keys to lists of modules. Each key corresponds to a specific route name
                (e.g., "text_query", "text_document", "image", "multimodal"). Each route contains a list of modules
                that will be applied sequentially when that route is selected.
            default_route: The default route to use if no task type or modality is specified. If None, an exception
                will be thrown if no task type is specified. If ``allow_empty_key`` is True, the first key in
                sub_modules will be used as the default route. Defaults to None.
            allow_empty_key: If True, allows the default route to be set to the first key in `sub_modules` if
                ``default_route`` is None. Defaults to True.
            route_mappings: Optional dictionary mapping (task, modality) tuples to route keys in sub_modules.
                This enables sophisticated routing logic based on combinations of task and modality:

                - Use ``None`` as a wildcard for either task or modality to create catch-all rules
                - Modality can be a string (e.g., ``"text"``, ``"image"``) or tuple (e.g., ``("text", "image")``)
                - Routes are resolved with a priority order (see **Route Resolution Priority** above)
                - All mapped routes must exist in ``sub_modules`` (validated at initialization)

                Example mappings::

                    {
                        # Exact matches (highest priority)
                        ("query", "text"): "efficient_text_encoder",
                        ("document", "text"): "powerful_text_encoder",

                        # Task with any modality
                        ("query", None): "query_encoder",  # All query tasks

                        # Any task with specific modality
                        (None, "image"): "image_encoder",  # All image inputs
                        (None, ("text", "image")): "multimodal_encoder",  # Multimodal inputs

                        # Catch-all (lowest priority)
                        (None, None): "default_encoder",
                    }

                If not provided, the router will attempt direct lookup using the task or modality as the route key
                in ``sub_modules``, then fall back to ``default_route``.
        """
        super().__init__()
        if sub_modules is None or len(sub_modules) == 0:
            raise ValueError("The routes dictionary cannot be empty.")
        if default_route is not None and default_route not in sub_modules:
            raise ValueError(f"Default route '{default_route}' not found in route keys: {list(sub_modules.keys())}")

        self.sub_modules = nn.ModuleDict(
            {route_name: nn.Sequential(*modules) for route_name, modules in sub_modules.items()}
        )

        # Validate that all route_mappings point to existing sub_modules
        if route_mappings:
            for (task, modality), target_route in route_mappings.items():
                if target_route not in sub_modules:
                    raise ValueError(
                        f"route_mappings contains mapping to '{target_route}' which is not in sub_modules. "
                        f"Available routes: {list(sub_modules.keys())}"
                    )

        # If allow_empty_key is True, we can set a default route to the first key in sub_modules.
        if allow_empty_key and default_route is None:
            default_route = next(iter(sub_modules.keys()))
        self.default_route = default_route
        self.allow_empty_key = allow_empty_key
        self.route_mappings = route_mappings or {}

    def _get_routes_string(self):
        # Build concise but helpful error message
        routes = [f"task={name!r}" for name in self.sub_modules.keys()]
        for task, modality in self.route_mappings.keys():
            if modality is None:
                routes.append(f"task={task!r}")
            elif task is None:
                routes.append(f"modality={modality!r}")
            else:
                routes.append(f"(task={task!r}, modality={modality!r})")
        if not routes:
            return ""
        elif len(routes) == 1:
            return routes[0]
        return ", ".join(routes[:-1]) + " and " + routes[-1]

    def _resolve_route_name(
        self, task: str | None = None, modality: str | tuple[str, ...] | None = None
    ) -> str | None:
        """
        Resolve the route key based on task and modality.

        Args:
            task: The task type (e.g., "query", "document")
            modality: The modality (e.g., "text", "image", ("text", "image"))

        Returns:
            The resolved route key, or None if no route is found
        """
        # Try exact match first: (task, modality)
        if (task, modality) in self.route_mappings:
            return self.route_mappings[(task, modality)]

        # Try task with any modality: (task, None)
        if (task, None) in self.route_mappings:
            return self.route_mappings[(task, None)]

        # Try any task with modality: (None, modality)
        if (None, modality) in self.route_mappings:
            return self.route_mappings[(None, modality)]

        # Try any task with any modality: (None, None)
        if (None, None) in self.route_mappings:
            return self.route_mappings[(None, None)]

        # Fallback: Try direct lookup by task
        if task and task in self.sub_modules:
            return task

        # Fallback: Try direct lookup by modality
        if modality and modality in self.sub_modules:
            return modality

        if task is not None:
            raise ValueError(f"No route found for task type '{task}'. Available routes: {self._get_routes_string()}")

        # No route found
        return None

    def _resolve_route(self, task: str | None = None, modality: str | tuple[str, ...] | None = None) -> str | None:
        route = self._resolve_route_name(task=task, modality=modality)

        # Fall back to default route if no route was found
        if route is None:
            route = self.default_route

        # If still no route, raise an error
        if route is None:
            if self.training:
                raise ValueError(
                    "You must provide a `router_mapping` argument on the training arguments, "
                    "or set a default route in the `Router` module."
                )

            error_msg = f"Could not determine route for task={task!r}, modality={modality!r}. "
            error_msg += f"Available routes: {self._get_routes_string()}. "
            error_msg += "Consider specifying the `task` parameter in `model.encode`, or setting a default route in the `Router`."
            raise ValueError(error_msg)

        if route not in self.sub_modules:
            raise ValueError(
                f"Resolved route '{route}' not found in sub_modules. Available submodule keys: {list(self.sub_modules.keys())}"
            )

        return route

    @classmethod
    def for_query_document(
        cls,
        query_modules: list[Module],
        document_modules: list[Module],
        default_route: str | None = "document",
        allow_empty_key: bool = True,
    ) -> Self:
        """
        Creates a Router model specifically for query and document modules, allowing convenient usage via `model.encode_query`
        and `model.encode_document`.

        Args:
            query_modules: List of modules to be applied for the "query" task type.
            document_modules: List of modules to be applied for the "document" task type.
            default_route: The default route to use if no task type is specified. If None, an exception will be thrown
                if no task type is specified. If ``allow_empty_key`` is True, the first key in sub_modules will be used as
                the default route. Defaults to "document".
            allow_empty_key: If True, allows the default route to be set to the first key in `sub_modules` if
                ``default_route`` is None. Defaults to True.

        Returns:
            Router: An instance of the Router model with the specified query and document modules.
        """
        return cls(
            sub_modules={"query": query_modules, "document": document_modules},
            default_route=default_route,
            allow_empty_key=allow_empty_key,
        )

    def forward(
        self,
        features: dict[str, Tensor],
        task: str | None = None,
        modality: str | tuple[str, ...] | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        # Get task from features if not provided
        if task is None:
            task = features.get("task", None)

        # Get modality from features if not provided
        if modality is None:
            modality = features.get("modality", None)

        # Resolve the route using task and modality
        route = self._resolve_route(task=task, modality=modality)

        # Pass task and modality to downstream modules
        kwargs["task"] = task
        kwargs["modality"] = modality

        for module in self.sub_modules[route]:
            module_kwargs = {
                key: value
                for key, value in kwargs.items()
                if hasattr(module, "forward_kwargs") and key in module.forward_kwargs
            }
            features = module(features, **module_kwargs)
        return features

    def get_sentence_embedding_dimension(self) -> int:
        for sub_modules in self.sub_modules.values():
            for module in reversed(sub_modules):
                if hasattr(module, "get_sentence_embedding_dimension"):
                    return module.get_sentence_embedding_dimension()
        return None

    def save(self, output_path: str, safe_serialization: bool = True, **kwargs):
        model_lookup = {}
        model_types = {}
        model_structure = {}

        for name, models in self.sub_modules.items():
            model_structure[name] = []
            for module_idx, model in enumerate(models):
                model_id = f"{name}_{module_idx}_{type(model).__name__}"
                model_lookup[model_id] = model
                model_types[model_id] = f"{type(model).__module__}.{type(model).__name__}"
                model_structure[name].append(model_id)

        for model_id, model in model_lookup.items():
            model_path = os.path.join(output_path, str(model_id))
            os.makedirs(model_path, exist_ok=True)
            try:
                model.save(model_path, safe_serialization=safe_serialization, **kwargs)
            except TypeError:
                # Fallback for legacy models that do not support kwargs
                model.save(model_path)

        # Get config dict and convert route_mappings keys to strings for JSON serialization
        config_dict = self.get_config_dict()
        if "route_mappings" in config_dict and config_dict["route_mappings"]:
            # Convert tuple keys to string representation for JSON
            config_dict["route_mappings"] = {str(key): value for key, value in config_dict["route_mappings"].items()}

        with open(os.path.join(output_path, self.config_file_name), "w", encoding="utf8") as fOut:
            json.dump(
                {
                    "types": model_types,
                    "structure": model_structure,
                    "parameters": config_dict,
                },
                fOut,
                indent=4,
            )

    def preprocess(
        self,
        texts: list[str] | list[tuple[str, str]],
        task: str | None = None,
        modality: Modality | None = None,
        **kwargs,
    ):
        """Preprocesses a text and maps tokens to token-ids"""
        # TODO: What if `texts` is an empty list

        # Backwards compatibility branch: for when texts are list of dicts with task types as keys
        if isinstance(texts[0], dict) and task is None:
            # Extract the task type key from the dictionaries
            dict_keys = set(key for text in texts for key in text.keys())
            # If the input keys are not modalities, they might be task types from backwards compatibility
            if not (dict_keys & set(MODALITY_TO_PROCESSOR_ARG.keys())):
                if len(dict_keys) > 1:
                    raise ValueError(
                        "You cannot pass a list of dictionaries with different task types. "
                        "Please ensure all dictionaries have the same task type key, or pass non-dictionary inputs "
                        "while providing the `task` argument."
                    )
                # If there's just one task type, extract it and remove the dictionary structure
                task = dict_keys.pop()
                texts = [text[task] for text in texts]

        # Infer modality if not provided
        """
        # TODO: Find an alternative for this
        if modality is None:
            try:
                modality = infer_modality(texts)
            except (ValueError, TypeError):
                # If modality inference fails, proceed without it
                pass
        """

        # Resolve route
        route = self._resolve_route(task=task, modality=modality)

        input_module = self.sub_modules[route][0]
        tokenized = input_module.preprocess(texts, **kwargs)
        tokenized["task"] = task
        if modality is not None:
            tokenized["modality"] = modality
        return tokenized

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        hub_kwargs = {
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        # Try the official config file first, then fall back to the legacy config file
        config = cls.load_config(model_name_or_path=model_name_or_path, subfolder=subfolder, **hub_kwargs)
        if not config:
            config = cls.load_config(
                model_name_or_path=model_name_or_path, config_filename="config.json", subfolder=subfolder, **hub_kwargs
            )
        modules = {}
        for model_id, model_type in config["types"].items():
            module_class: Module = import_from_string(model_type)
            try:
                module = module_class.load(
                    model_name_or_path, subfolder=Path(subfolder, model_id).as_posix(), **hub_kwargs, **kwargs
                )
            except TypeError:
                local_path = load_dir_path(
                    model_name_or_path=model_name_or_path, subfolder=Path(subfolder, model_id).as_posix(), **hub_kwargs
                )
                module = module_class.load(local_path)
            modules[model_id] = module

        model_structure = {}
        for key_name, models_list in config["structure"].items():
            model_structure[key_name] = []
            for model_id in models_list:
                model_structure[key_name].append(modules[model_id])

        # Convert route_mappings string keys back to tuples
        parameters = config["parameters"].copy()
        if "route_mappings" in parameters and parameters["route_mappings"]:
            route_mappings = {}
            for key_str, value in parameters["route_mappings"].items():
                # Parse the string representation back to tuple
                # Format: "('task', 'modality')" or "(None, 'modality')" etc.
                try:
                    import ast

                    key_tuple = ast.literal_eval(key_str)
                    route_mappings[key_tuple] = value
                except (ValueError, SyntaxError):
                    logger.warning(f"Could not parse route_mapping key: {key_str}. Skipping.")
            parameters["route_mappings"] = route_mappings

        model = cls(model_structure, **parameters)
        return model

    @property
    def tokenizer(self):
        # We might have multiple tokenizers, one for each route, but we can only return one here.
        for sub_modules in self.sub_modules.values():
            input_module: InputModule = sub_modules[0]
            if hasattr(input_module, "tokenizer") and input_module.tokenizer is not None:
                return input_module.tokenizer
        return None

    @property
    def max_seq_length(self) -> int:
        # Collect all unique max_seq_length values
        max_seq_lengths = set()
        for modules in self.sub_modules.values():
            input_module: InputModule = modules[0]
            if modules and hasattr(input_module, "max_seq_length"):
                if max_seq_length := input_module.max_seq_length:
                    max_seq_lengths.add(max_seq_length)

        if not max_seq_lengths:
            return None
        elif len(max_seq_lengths) == 1:
            # Only one unique max_seq_length
            return max_seq_lengths.pop()
        else:
            logger.warning_once(f"Different max_seq_lengths detected: {max_seq_lengths}. Using the maximum value.")
            return max(max_seq_lengths)

    @max_seq_length.setter
    def max_seq_length(self, value) -> None:
        # Check which modules have max_seq_length
        has_max_seq_length_keys = []
        for key, models in self.sub_modules.items():
            if models and hasattr(models[0], "max_seq_length"):
                has_max_seq_length_keys.append(key)

        if len(has_max_seq_length_keys) == 0:
            logger.warning("No modules have a max_seq_length attribute to set.")
            return

        for key in has_max_seq_length_keys:
            input_module: InputModule = self.sub_modules[key][0]
            input_module.max_seq_length = value


# For backwards compatibility, we ensure that the legacy `Asym` alias points to the new `Router` class.
Asym = Router

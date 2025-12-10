from __future__ import annotations

import hashlib
import logging
import os
import random
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from tqdm import trange
from tqdm.autonotebook import tqdm

from .environment import is_datasets_available

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datasets import Dataset

    from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
    from sentence_transformers.SentenceTransformer import SentenceTransformer


def mine_hard_negatives(
    dataset: Dataset,
    model: SentenceTransformer,
    anchor_column_name: str | None = None,
    positive_column_name: str | None = None,
    corpus: list[str] | None = None,
    cross_encoder: CrossEncoder | None = None,
    range_min: int = 0,
    range_max: int | None = None,
    max_score: float | None = None,
    min_score: float | None = None,
    absolute_margin: float | None = None,
    relative_margin: float | None = None,
    num_negatives: int = 3,
    sampling_strategy: Literal["random", "top"] = "top",
    query_prompt_name: str | None = None,
    query_prompt: str | None = None,
    corpus_prompt_name: str | None = None,
    corpus_prompt: str | None = None,
    include_positives: bool = False,
    output_format: Literal["triplet", "n-tuple", "labeled-pair", "labeled-list"] = "triplet",
    output_scores: bool = False,
    batch_size: int = 32,
    faiss_batch_size: int = 16384,
    use_faiss: bool = False,
    use_multi_process: list[str] | bool = False,
    verbose: bool = True,
    cache_folder: str | None = None,
    as_triplets: bool | None = None,
    margin: float | None = None,
) -> Dataset:
    """
    Add hard negatives to a dataset of (anchor, positive) pairs to create (anchor, positive, negative) triplets or
    (anchor, positive, negative_1, ..., negative_n) tuples.

    Hard negative mining is a technique to improve the quality of a dataset by adding hard negatives, which are
    texts that may appear similar to the anchor, but are not. Using hard negatives can improve the performance of
    models trained on the dataset.

    This function uses a SentenceTransformer model to embed the sentences in the dataset, and then finds the closest
    matches to each anchor sentence in the dataset. It then samples negatives from the closest matches, optionally
    using a CrossEncoder model to rescore the candidates.

    Supports prompt formatting for models that expect specific instruction-style input.

    You can influence the candidate negative selection in various ways:

    - **range_min**: Minimum rank of the closest matches to consider as negatives: useful to skip the most similar texts to
      avoid marking texts as negative that are actually positives.
    - **range_max**: Maximum rank of the closest matches to consider as negatives: useful to limit the number of candidates
      to sample negatives from. A lower value makes processing faster, but may result in less candidate negatives that
      satisfy the margin or max_score conditions.
    - **max_score**: Maximum score to consider as a negative: useful to skip candidates that are too similar to the anchor.
    - **min_score**: Minimum score to consider as a negative: useful to skip candidates that are too dissimilar to the anchor.
    - **absolute_margin**: Absolute margin for hard negative mining: useful to skip candidate negatives whose similarity
      to the anchor is within a certain margin of the positive pair. A value of 0 can be used to enforce that the negative
      is always further away from the anchor than the positive.
    - **relative_margin**: Relative margin for hard negative mining: useful to skip candidate negatives whose similarity
      to the anchor is within a certain margin of the positive pair. A value of 0.05 means that the negative is at most 95%
      as similar to the anchor as the positive.
    - **sampling_strategy**: Sampling strategy for negatives: "top" or "random". "top" will always sample the top n
      candidates as negatives, while "random" will sample n negatives randomly from the candidates that satisfy the
      margin or max_score conditions.

    .. tip::

        The excellent `NV-Retriever paper <https://huggingface.co/papers/2407.15831>`_ is a great resource for understanding the
        details of hard negative mining and how to use it effectively. Notably, it reaches the strongest performance using
        these settings::

            dataset = mine_hard_negatives(
                dataset=dataset,
                model=model,
                relative_margin=0.05,         # 0.05 means that the negative is at most 95% as similar to the anchor as the positive
                num_negatives=num_negatives,  # 10 or less is recommended
                sampling_strategy="top",      # "top" means that we sample the top candidates as negatives
                batch_size=batch_size,        # Adjust as needed
                use_faiss=True,               # Optional: Use faiss/faiss-gpu for faster similarity search
            )

        This corresponds with the `TopK-PercPos (95%)` mining method.

    Example:

        >>> from sentence_transformers.util import mine_hard_negatives
        >>> from sentence_transformers import SentenceTransformer
        >>> from datasets import load_dataset
        >>> # Load a Sentence Transformer model
        >>> model = SentenceTransformer("all-MiniLM-L6-v2")
        >>>
        >>> # Load a dataset to mine hard negatives from
        >>> dataset = load_dataset("sentence-transformers/natural-questions", split="train")
        >>> dataset
        Dataset({
            features: ['query', 'answer'],
            num_rows: 100231
        })
        >>> dataset = mine_hard_negatives(
        ...     dataset=dataset,
        ...     model=model,
        ...     range_min=10,
        ...     range_max=50,
        ...     max_score=0.8,
        ...     relative_margin=0.05,
        ...     num_negatives=5,
        ...     sampling_strategy="random",
        ...     batch_size=128,
        ...     use_faiss=True,
        ... )
        Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 588/588 [00:32<00:00, 18.07it/s]
        Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 784/784 [00:08<00:00, 96.41it/s]
        Querying FAISS index: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:06<00:00,  1.06it/s]
        Negative candidates mined, preparing dataset...
        Metric       Positive       Negative     Difference
        Count         100,231        487,865
        Mean           0.6866         0.4194         0.2752
        Median         0.7010         0.4102         0.2760
        Std            0.1125         0.0719         0.1136
        Min            0.0303         0.1702         0.0209
        25%            0.6221         0.3672         0.1899
        50%            0.7010         0.4102         0.2760
        75%            0.7667         0.4647         0.3590
        Max            0.9584         0.7621         0.7073
        Skipped 427,503 potential negatives (8.36%) due to the relative_margin of 0.05.
        Skipped 978 potential negatives (0.02%) due to the max_score of 0.8.
        Could not find enough negatives for 13290 samples (2.65%). Consider adjusting the range_max, range_min, relative_margin and max_score parameters if you'd like to find more valid negatives.
        >>> dataset
        Dataset({
            features: ['query', 'answer', 'negative'],
            num_rows: 487865
        })
        >>> dataset[0]
        {
            'query': 'when did richmond last play in a preliminary final',
            'answer': "Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieved since 1995. A series of close losses hampered the Tigers throughout the middle of the season, including a 5-point loss to the Western Bulldogs, 2-point loss to Fremantle, and a 3-point loss to the Giants. Richmond ended the season strongly with convincing victories over Fremantle and St Kilda in the final two rounds, elevating the club to 3rd on the ladder. Richmond's first final of the season against the Cats at the MCG attracted a record qualifying final crowd of 95,028; the Tigers won by 51 points. Having advanced to the first preliminary finals for the first time since 2001, Richmond defeated Greater Western Sydney by 36 points in front of a crowd of 94,258 to progress to the Grand Final against Adelaide, their first Grand Final appearance since 1982. The attendance was 100,021, the largest crowd to a grand final since 1986. The Crows led at quarter time and led by as many as 13, but the Tigers took over the game as it progressed and scored seven straight goals at one point. They eventually would win by 48 points – 16.12 (108) to Adelaide's 8.12 (60) – to end their 37-year flag drought.[22] Dustin Martin also became the first player to win a Premiership medal, the Brownlow Medal and the Norm Smith Medal in the same season, while Damien Hardwick was named AFL Coaches Association Coach of the Year. Richmond's jump from 13th to premiers also marked the biggest jump from one AFL season to the next.",
            'negative': "2018 NRL Grand Final The 2018 NRL Grand Final was the conclusive and premiership-deciding game of the 2018 National Rugby League season and was played on Sunday September 30 at Sydney's ANZ Stadium.[1] The match was contested between minor premiers the Sydney Roosters and defending premiers the Melbourne Storm. In front of a crowd of 82,688, Sydney won the match 21â€“6 to claim their 14th premiership title and their first since 2013. Roosters five-eighth Luke Keary was awarded the Clive Churchill Medal as the game's official man of the match."
        }
        >>> # To include similarity scores, use output_scores=True
        >>> dataset_with_scores = mine_hard_negatives(
        ...     dataset=dataset,
        ...     model=model,
        ...     output_scores=True,
        ...     # ... other parameters
        ... )
        >>> dataset_with_scores
        Dataset({
            features: ['query', 'answer', 'negative', 'scores'],
            num_rows: 487865
        })
        >>> dataset.push_to_hub("natural-questions-hard-negatives", "triplet-all")

    Args:
        dataset (Dataset): A dataset containing (anchor, positive) pairs.
        model (SentenceTransformer): A SentenceTransformer model to use for embedding the sentences.
        anchor_column_name (str, optional): The column name in `dataset` that contains the anchor/query. Defaults to None, in which case the first column in `dataset` will be used.
        positive_column_name (str, optional): The column name in `dataset` that contains the positive candidates. Defaults to None, in which case the second column in `dataset` will be used.
        corpus (List[str], optional): A list containing documents as strings that will be used as candidate negatives
            in addition to the second column in `dataset`. Defaults to None, in which case the second column in
            `dataset` will exclusively be used as the negative candidate corpus.
        cross_encoder (CrossEncoder, optional): A CrossEncoder model to use for rescoring the candidates. Defaults to None.
        range_min (int): Minimum rank of the closest matches to consider as negatives. Defaults to 0.
        range_max (int, optional): Maximum rank of the closest matches to consider as negatives. Defaults to None.
        max_score (float, optional): Maximum score to consider as a negative. Defaults to None.
        min_score (float, optional): Minimum score to consider as a negative. Defaults to None.
        absolute_margin (float, optional): Absolute margin for hard negative mining, i.e. the minimum distance between
            the positive similarity and the negative similarity. Defaults to None.
        relative_margin (float, optional): Relative margin for hard negative mining, i.e. the maximum ratio between
            the positive similarity and the negative similarity. A value of 0.05 means that the negative is at most
            95% as similar to the anchor as the positive. Defaults to None.
        num_negatives (int): Number of negatives to sample. Defaults to 3.
        sampling_strategy (Literal["random", "top"]): Sampling strategy for negatives: "top" or "random". Defaults to "top".
        query_prompt_name (Optional[str], optional): The name of a predefined prompt to use when encoding the first/anchor dataset column.
            It must match a key in the ``model.prompts`` dictionary, which can be set during model initialization
            or loaded from the model configuration.

            For example, if ``query_prompt_name="query"`` and the model prompts dictionary includes {"query": "query: "},
            then the sentence "What is the capital of France?" is transformed into: "query: What is the capital of France?"
            before encoding. This is useful for models that were trained or fine-tuned with specific prompt formats.

            Ignored if ``query_prompt`` is provided. Defaults to None.

        query_prompt (Optional[str], optional): A raw prompt string to prepend directly to the first/anchor dataset column during encoding.

            For instance, `query_prompt="query: "` transforms the sentence "What is the capital of France?" into:
            "query: What is the capital of France?". Use this to override the prompt logic entirely and supply your own prefix.
            This takes precedence over ``query_prompt_name``. Defaults to None.
        corpus_prompt_name (Optional[str], optional): The name of a predefined prompt to use when encoding the corpus. See
            ``query_prompt_name`` for more information. Defaults to None.
        corpus_prompt (Optional[str], optional): A raw prompt string to prepend directly to the corpus during encoding.
            See ``query_prompt`` for more information. Defaults to None.
        include_positives (bool): Whether to include the positives in the negative candidates.
            Setting this to True is primarily useful for creating Reranking evaluation datasets for CrossEncoder models,
            where it can be useful to get a full ranking (including the positives) from a first-stage retrieval model.
            Defaults to False.
        output_format (Literal["triplet", "n-tuple", "labeled-pair", "labeled-list"]): Output format for the `datasets.Dataset`.
            When ``output_scores=False`` (default), options are:

            - "triplet": (anchor, positive, negative) triplets, i.e. 3 columns. Useful for e.g. :class:`~sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss`.
            - "n-tuple": (anchor, positive, negative_1, ..., negative_n) tuples, i.e. 2 + num_negatives columns. Useful for e.g. :class:`~sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss`.
            - "labeled-pair": (anchor, passage, label) text tuples with a label of 0 for negative and 1 for positive, i.e. 3 columns. Useful for e.g. :class:`~sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss`.
            - "labeled-list": (anchor, [doc1, doc2, ..., docN], [label1, label2, ..., labelN]) tuples with labels of 0 for negative and 1 for positive, i.e. 3 columns. Useful for e.g. :class:`~sentence_transformers.cross_encoder.losses.LambdaLoss`.

            Defaults to "triplet". See ``output_scores`` for the output formats when ``output_scores=True``.
        output_scores (bool): Whether to include similarity scores in the output dataset. When True, adds score fields to the output:
            - For "triplet" format: adds `scores` column with query-positive and query-negative similarity scores, for 4 columns total.
            - For "n-tuple" format: adds `scores` column with a list of similarity scores for the query-positive and each of the query-negative pairs, for 3 + num_negatives columns total. Useful for e.g. :class:`~sentence_transformers.sparse_encoder.losses.SparseMarginMSELoss`.
            - For "labeled-pair" format: replaces the `label` column with a `score` column. Labels are binary (1 for positive, 0 for negative), but scores contain the actual similarity scores computed by the model or cross_encoder. The output has 3 columns.
            - For "labeled-list" format: replaces the `labels` column with a `scores` column. Labels are binary (1 for positive, 0 for negative), but scores contain the actual similarity scores computed by the model or cross_encoder. The output has 3 columns.
            Defaults to False.
        batch_size (int): Batch size for encoding the dataset. Defaults to 32.
        faiss_batch_size (int): Batch size for FAISS top-k search. Defaults to 16384.
        use_faiss (bool): Whether to use FAISS for similarity search. May be recommended for large datasets. Defaults to False.
        use_multi_process (bool | List[str], optional): Whether to use multi-GPU/CPU processing. If True, uses all GPUs if CUDA
            is available, and 4 CPU processes if it's not available. You can also pass a list of PyTorch devices like
            ["cuda:0", "cuda:1", ...] or ["cpu", "cpu", "cpu", "cpu"].
        verbose (bool): Whether to print statistics and logging. Defaults to True.
        cache_folder (str, optional): Directory path for caching embeddings. If provided, the function will save
            ``query_embeddings_{hash}.npy`` and ``corpus_embeddings_{hash}.npy`` under this folder after the first run,
            and on subsequent calls will load from these files if they exist to avoid recomputation. The hashes are
            computed based on the model name and the queries/corpus. Defaults to None.
        as_triplets (bool, optional): Deprecated. Use `output_format` instead. Defaults to None.
        margin (float, optional): Deprecated. Use `absolute_margin` or `relative_margin` instead. Defaults to None.

    Returns:
        Dataset: A dataset containing the specified output format. If `output_scores=False` (default), the formats are:

        - "triplet": (anchor, positive, negative)
        - "n-tuple": (anchor, positive, negative_1, ..., negative_n)
        - "labeled-pair": (anchor, passage, label)
        - "labeled-list": (anchor, [passages], [labels])

        And if `output_scores=True`, the formats are:

        - "triplet": (anchor, positive, negative, [scores])
        - "n-tuple": (anchor, positive, negative_1, ..., negative_n, [scores])
        - "labeled-pair": (anchor, passage, score)
        - "labeled-list": (anchor, [passages], [scores])
    """
    if not is_datasets_available():
        raise ImportError("Please install `datasets` to use this function: `pip install datasets`.")

    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Please provide a non-empty dataset.")

    from datasets import Dataset

    # If a dataset has duplicate queries, assume that all duplicates are positive pairs.
    columns = dataset.column_names

    if not anchor_column_name or anchor_column_name not in columns:
        anchor_column_name = columns[0]

    if not positive_column_name or positive_column_name not in columns:
        positive_column_name = columns[1]

    if not anchor_column_name and not positive_column_name and len(columns) != 2:
        raise ValueError("Dataset must contain exactly two columns.")

    if as_triplets is not None:
        output_format = "triplet" if as_triplets else "n-tuple"
        logger.warning(
            "The `as_triplets` parameter is deprecated. Use the `output_format` parameter instead. "
            f"Setting `output_format` to `{output_format}`."
        )

    if include_positives:
        if (
            range_min != 0
            or range_max is not None
            or max_score is not None
            or margin is not None
            or sampling_strategy != "top"
        ):
            logger.warning(
                "When using `include_positives=True`, updating `range_min`, `range_max`, `max_score`, `margin`, or "
                "`sampling_strategy` from the default values may still discard the positive values."
            )
        if output_format != "n-tuple":
            logger.warning(
                'When using `include_positives=True`, `output_format` will be set to `"n-tuple"` to ensure that the ranking order is preserved.'
            )
            output_format = "n-tuple"

    if output_format == "n-tuple-scores":
        logger.warning(
            '`output_format="n-tuple-scores"` is deprecated. Use `output_format="n-tuple"` with '
            "`output_scores=True` instead for equivalent output formats."
        )
        output_format = "n-tuple"
        output_scores = True

    if output_format not in ["triplet", "n-tuple", "labeled-pair", "labeled-list"]:
        raise ValueError(
            f"Invalid output_format: {output_format}. Must be one of 'triplet', 'n-tuple', 'labeled-pair', or 'labeled-list'."
        )

    if margin is not None:
        absolute_margin = margin
        logger.warning(
            "The `margin` parameter is deprecated. Use the `absolute_margin` and/or `relative_margin` parameter instead. "
            f"Setting `absolute_margin` to `{absolute_margin}`."
        )

    # To avoid re-embedding the same query multiple times, we keep a counter of the number of positives per query
    positives_per_query = list(
        dataset.to_pandas().groupby(anchor_column_name).count().to_dict()[positive_column_name].values()
    )
    max_positives = max(positives_per_query)

    if range_max is None:
        if absolute_margin is not None or relative_margin is not None or max_score is not None:
            # max_positives + 10 * num_negatives negatives because some might be skipped, and range_min skipped
            range_max = range_min + (num_negatives * 10) + max_positives
        else:
            # max_positives, num_negatives negatives, and range_min skipped
            range_max = range_min + num_negatives + max_positives
        if range_max > 2048 and use_faiss:
            # FAISS on GPU can only retrieve up to 2048 documents per query
            range_max = 2048
            if verbose:
                print("Using FAISS, we can only retrieve up to 2048 documents per query. Setting range_max to 2048.")
        if verbose:
            print(f"Setting range_max to {range_max} based on the provided parameters.")

    log_counters = {}
    queries = list(dataset[anchor_column_name])
    positives = list(dataset[positive_column_name])
    separate_corpus = corpus is not None
    if not separate_corpus:
        corpus = positives

    # Deduplicate the corpus
    # make sure all the positives are also in the corpus and de-duplicate it.
    corpus = list(dict.fromkeys(corpus + positives))

    # corpus_idx maps the corpus text into its position in the corpus
    # This position does not necessarily matches the original corpus, as it was de-duplicated.
    corpus_idx = {text: idx for idx, text in enumerate(corpus)}

    # Deduplicate the queries, but keep the original one for later reference.
    all_queries = queries.copy()
    queries = list(dict.fromkeys(queries))
    queries_idx = {query: idx for idx, query in enumerate(queries)}
    n_queries = len(queries)
    batch_idx = torch.arange(n_queries).unsqueeze(-1)

    device = model.device

    if n_queries != len(all_queries) and verbose:
        print(f"Found {n_queries} unique queries out of {len(all_queries)} total queries.")

    if max_positives > 1:
        avg_positives_per_query = np.mean(positives_per_query)
        print(f"Found an average of {avg_positives_per_query:.3f} positives per query.")

    corpus_embeddings = None
    query_embeddings = None

    if cache_folder:
        os.makedirs(cache_folder, exist_ok=True)

        model_name = model.model_card_data.base_model or ""
        query_hash = hashlib.sha256((model_name + "".join(queries)).encode(), usedforsecurity=False).hexdigest()
        corpus_hash = hashlib.sha256((model_name + "".join(corpus)).encode(), usedforsecurity=False).hexdigest()

        query_cache_file = os.path.join(cache_folder, f"query_embeddings_{query_hash}.npy")
        corpus_cache_file = os.path.join(cache_folder, f"corpus_embeddings_{corpus_hash}.npy")

        if os.path.exists(query_cache_file):
            query_embeddings = np.load(query_cache_file)
            if verbose:
                print(f"[Cache] Loaded query embeddings from {query_cache_file} (shape={query_embeddings.shape})")

        if os.path.exists(corpus_cache_file):
            corpus_embeddings = np.load(corpus_cache_file)
            if verbose:
                print(f"[Cache] Loaded corpus embeddings from {corpus_cache_file} (shape={corpus_embeddings.shape})")

    # Embed the corpus and the queries
    if corpus_embeddings is None or query_embeddings is None:
        if use_multi_process:
            pool = model.start_multi_process_pool(
                target_devices=None if isinstance(use_multi_process, bool) else use_multi_process
            )
        else:
            pool = None
        if corpus_embeddings is None:
            corpus_embeddings = model.encode_document(
                corpus,
                pool=pool,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
                prompt_name=corpus_prompt_name,
                prompt=corpus_prompt,
            )
        if query_embeddings is None:
            query_embeddings = model.encode_query(
                queries,
                pool=pool,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
                prompt_name=query_prompt_name,
                prompt=query_prompt,
            )
        if use_multi_process:
            model.stop_multi_process_pool(pool)

    if cache_folder:
        if not os.path.exists(query_cache_file):
            np.save(query_cache_file, query_embeddings)
            if verbose:
                print(f"[Cache] Saved query embeddings to {query_cache_file}")

        if not os.path.exists(corpus_cache_file):
            np.save(corpus_cache_file, corpus_embeddings)
            if verbose:
                print(f"[Cache] Saved corpus embeddings to {corpus_cache_file}")

    if use_faiss:
        import faiss

        index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
        # Move the index to the GPU if available
        try:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            index: faiss.IndexFlatIP = faiss.index_cpu_to_all_gpus(index, co=co)
        except Exception:
            pass

        index.add(corpus_embeddings)

        scores_list = []
        indices_list = []
        # Iterate over query embeddings in batches so we can track the progress
        for i in trange(0, len(query_embeddings), faiss_batch_size, desc="Querying FAISS index"):
            query_chunk = query_embeddings[i : i + faiss_batch_size]
            scores, indices = index.search(query_chunk, k=range_max + 1)
            scores_list.append(scores)
            indices_list.append(indices)
        scores = torch.from_numpy(np.concatenate(scores_list, axis=0)).to(device)
        indices = torch.from_numpy(np.concatenate(indices_list, axis=0)).to(device)

    else:
        # Compute the similarity scores between the queries and the corpus
        scores = model.similarity(query_embeddings, corpus_embeddings).to(device)

        # Keep only the range_max + max_positives highest scores. We offset by 1 to potentially include the positive pair
        scores, indices = torch.topk(scores, k=range_max + max_positives, dim=1)

    # As we may have duplicated queries (i.e., a single query with multiple positives),
    # We keep track, for each unique query, of where their positives are in the list of positives (positive_indices).
    # Note that as queries may have differing numbers of positives, we cannot guarantee that this is a fixed-length matrix.
    positive_indices = [[] for _ in range(n_queries)]

    for query, positive in zip(all_queries, positives):
        query_idx = queries_idx[query]
        if corpus_idx[positive] not in positive_indices[query_idx]:
            positive_indices[query_idx].append(corpus_idx[positive])

    n_positives = [len(p) for p in positive_indices]

    # Reorder the positives and all_queries according to the deduplicated queries
    positives = []
    all_queries = []
    for idx in range(n_queries):
        positives.extend([corpus[doc_idx] for doc_idx in positive_indices[idx]])
        all_queries.extend([queries[idx]] * n_positives[idx])

    positive_indices = [torch.tensor(p, device=device) for p in positive_indices]

    # Compute the positive scores
    query_embeddings = query_embeddings[[idx for idx in range(n_queries) for _ in range(n_positives[idx])]]
    positive_embeddings = corpus_embeddings[torch.cat(positive_indices).tolist()]
    positive_scores = model.similarity_pairwise(query_embeddings, positive_embeddings).to(device)

    del query_embeddings
    del positive_embeddings
    del corpus_embeddings

    # Rescore with cross_encoder
    if cross_encoder is not None and (
        absolute_margin is not None or relative_margin is not None or max_score is not None
    ):
        if use_multi_process:
            pool = cross_encoder.start_multi_process_pool(
                target_devices=None if isinstance(use_multi_process, bool) else use_multi_process
            )
        else:
            pool = None
        for idx, candidate_idx in tqdm(enumerate(indices), desc="Rescoring with CrossEncoder", total=len(indices)):
            query = queries[idx]
            candidate_passages = [corpus[_idx] for _idx in candidate_idx]
            num_candidates = len(candidate_passages)
            pred_scores = cross_encoder.predict(
                list(zip([query] * num_candidates, candidate_passages)),
                batch_size=batch_size,
                convert_to_tensor=True,
                pool=pool,
            ).to(device)
            scores[idx] = pred_scores
        positive_scores = cross_encoder.predict(
            list(zip(all_queries, positives)),
            batch_size=batch_size,
            convert_to_tensor=True,
            pool=pool,
        ).to(device)
        if use_multi_process:
            cross_encoder.stop_multi_process_pool(pool)

    if not include_positives:
        # for each query, create a mask that is True for the positives and False for the negatives in the indices
        positive_mask = torch.stack(
            [torch.isin(indices[q_idx], positive_indices[q_idx]) for q_idx in range(n_queries)]
        )

        # Scores is a [num_queries, range_max] tensor, where we set the values to -inf to disqualify the corresponding
        # positive candidates
        scores[positive_mask] = -float("inf")

    num_candidates = scores.numel()

    # Remove based on margin
    if absolute_margin is not None or relative_margin is not None:
        # If we have a margin, we will remove candidates that are too close to the positive pair
        # If there are multiple positives, we need to define which one to use for the margin
        # To be on the safe side, we will use the _minimum_ positive score (i.e., harder positive) for the margin
        max_positive_scores = torch.empty(n_queries, device=positive_scores.device, dtype=positive_scores.dtype)
        positive_score_idx = 0
        for q_idx in range(n_queries):
            max_positive_scores[q_idx] = torch.min(
                positive_scores[positive_score_idx : positive_score_idx + n_positives[q_idx]]
            )
            positive_score_idx += n_positives[q_idx]

        if absolute_margin is not None:
            removed_indices = scores + absolute_margin > max_positive_scores.repeat(scores.size(1), 1).T
            scores[removed_indices] = -float("inf")

            num_skipped = removed_indices.sum().item()
            if num_skipped:
                log_counters["absolute_margin"] = {"skipped": num_skipped, "ratio": num_skipped / num_candidates}
                num_candidates -= num_skipped

        if relative_margin is not None:
            removed_indices = scores > max_positive_scores.repeat(scores.size(1), 1).T * (1 - relative_margin)
            scores[removed_indices] = -float("inf")

            num_skipped = removed_indices.sum().item()
            if num_skipped:
                log_counters["relative_margin"] = {"skipped": num_skipped, "ratio": num_skipped / num_candidates}
                num_candidates -= num_skipped

    # Remove based on max_score
    if max_score is not None:
        removed_indices = scores > max_score
        scores[removed_indices] = -float("inf")

        num_skipped = removed_indices.sum().item()
        if num_skipped:
            log_counters["max_score"] = {
                "skipped": num_skipped,
                "ratio": num_skipped / num_candidates,
            }

    # Remove based on min_score
    if min_score is not None:
        removed_indices = scores < min_score
        scores[removed_indices] = -float("inf")

        num_skipped = removed_indices.sum().item()
        if num_skipped:
            log_counters["min_score"] = {
                "skipped": num_skipped,
                "ratio": num_skipped / num_candidates,
            }

    # Grab the top negative candidates and remove the first range_min candidates
    negative_scores, local_indices = torch.topk(scores, k=range_max, dim=1)
    indices = indices[batch_idx, local_indices]

    if range_min:
        indices = indices[:, range_min:]
        negative_scores = negative_scores[:, range_min:]

    # Either grab the top negatives or sample randomly
    if sampling_strategy == "top":
        indices = indices[:, :num_negatives]
        negative_scores = negative_scores[:, :num_negatives]

    elif sampling_strategy == "random":
        # Prevent sampling -inf values if possible
        num_options = indices.size(1) - negative_scores.isinf().sum(1)
        num_options = num_options.clamp(min=num_negatives)
        # Randomly sample negatives from each row
        sampled_idx = [random.sample(range(options), k=num_negatives) for options in num_options]
        indices = indices[batch_idx, sampled_idx]
        negative_scores = negative_scores[batch_idx, sampled_idx]
        # Resort the indices and scores
        negative_scores, local_indices = negative_scores.sort(dim=1, descending=True)
        indices = indices[batch_idx, local_indices]

    if verbose:
        print("Negative candidates mined, preparing dataset...")

    if output_format == "triplet":
        # repeat indices and negative_scores by the number of positives of each query
        indices = torch.cat([indices[idx].repeat(n_positives[idx], 1) for idx in range(n_queries)])
        negative_scores = torch.cat([negative_scores[idx].repeat(n_positives[idx], 1) for idx in range(n_queries)])

        # If calling as triples and there are multiple positives per query, we will explode the dataset into triplets.
        indices_to_keep = negative_scores != -float("inf")
        anchor_indices = torch.empty_like(indices)
        pos_indices = torch.empty_like(indices)

        indices = indices[indices_to_keep]
        negative_scores = negative_scores[indices_to_keep]

        # the anchor_indices matrix is shaped [n_total_queries, n_negatives]
        start_idx = 0
        for q_idx in range(n_queries):
            anchor_indices[start_idx : start_idx + n_positives[q_idx]] = torch.tensor(q_idx).repeat(
                n_positives[q_idx], num_negatives
            )
            pos_indices[start_idx : start_idx + n_positives[q_idx]] = (
                positive_indices[q_idx].repeat(num_negatives, 1).T
            )
            start_idx += n_positives[q_idx]

        anchor_indices = anchor_indices[indices_to_keep]
        positive_indices = pos_indices[indices_to_keep]

        # Positive scores aligned per (anchor, positive, negative) triplet
        positive_triplet_scores = positive_scores.repeat(num_negatives, 1).T[indices_to_keep]

        # Base textual fields
        dataset_data = {
            anchor_column_name: [],
            positive_column_name: [],
            "negative": [],
        }

        if output_scores:
            # Single `scores` field containing [pos_score, neg_score]
            dataset_data["scores"] = []

        for anchor_idx, positive_idx, negative_idx, pos_score, neg_score in zip(
            anchor_indices, positive_indices, indices, positive_triplet_scores, negative_scores
        ):
            dataset_data[anchor_column_name].append(queries[anchor_idx])
            dataset_data[positive_column_name].append(corpus[positive_idx])
            dataset_data["negative"].append(corpus[negative_idx])
            if output_scores:
                dataset_data["scores"].append([pos_score.item(), neg_score.item()])
        difference_scores = positive_triplet_scores - negative_scores
        maximum_possible_samples = indices_to_keep.numel()

    elif output_format == "labeled-pair":
        # In labeled-pair format we either expose a binary `label` or a
        # single continuous `score` column, but never both at once.
        dataset_data = {
            anchor_column_name: [],
            positive_column_name: [],  # Note, this is not strictly positives
        }
        if output_scores:
            dataset_data["score"] = []
        else:
            dataset_data["label"] = []

        positive_score_idx = 0
        for query_idx in range(n_queries):
            for positive_idx in positive_indices[query_idx]:
                dataset_data[anchor_column_name].append(queries[query_idx])
                dataset_data[positive_column_name].append(corpus[positive_idx])
                if output_scores:
                    pos_score = positive_scores[positive_score_idx].item()
                    dataset_data["score"].append(pos_score)
                else:
                    dataset_data["label"].append(1)
                positive_score_idx += 1

            for negative_idx, negative_score in zip(indices[query_idx], negative_scores[query_idx]):
                if negative_score == -float("inf"):
                    continue
                dataset_data[anchor_column_name].append(queries[query_idx])
                dataset_data[positive_column_name].append(corpus[negative_idx])
                if output_scores:
                    dataset_data["score"].append(negative_score.item())
                else:
                    dataset_data["label"].append(0)

        negative_scores = torch.cat([negative_scores[idx].repeat(n_positives[idx], 1) for idx in range(n_queries)])
        indices_to_keep = negative_scores != -float("inf")
        negative_scores = negative_scores[indices_to_keep]
        difference_scores = positive_scores.repeat(num_negatives, 1).T[indices_to_keep] - negative_scores
        maximum_possible_samples = n_queries * num_negatives + len(dataset)

    elif output_format == "n-tuple":
        # repeat indices and negative_scores by the number of positives of each query
        indices = torch.cat([indices[idx].repeat(n_positives[idx], 1) for idx in range(n_queries)])
        negative_scores = torch.cat([negative_scores[idx].repeat(n_positives[idx], 1) for idx in range(n_queries)])

        # Keep only indices where num_negative negatives were found
        indices_to_keep = (negative_scores != -float("inf")).all(dim=1)
        negative_scores = negative_scores[indices_to_keep]
        indices = indices[indices_to_keep]

        # Base textual fields: anchor, positive and N negatives.
        dataset_data = {
            anchor_column_name: [all_queries[idx] for idx, keep in enumerate(indices_to_keep) if keep],
            positive_column_name: [positives[idx] for idx, keep in enumerate(indices_to_keep) if keep],
            **{
                f"negative_{i}": [corpus[neg_idx] for neg_idx in neg_indices]
                for i, neg_indices in enumerate(indices.T, start=1)
            },
        }
        if output_scores:
            # Single `scores` field containing [pos_score, neg_1_score, ..., neg_n_score]
            dataset_data["scores"] = torch.cat(
                [positive_scores[indices_to_keep].unsqueeze(-1), negative_scores], dim=1
            ).tolist()
        negative_scores = negative_scores.flatten()
        difference_scores = positive_scores.repeat(num_negatives, 1).T[indices_to_keep].flatten() - negative_scores
        maximum_possible_samples = indices_to_keep.size(0)

    elif output_format == "labeled-list":
        # repeat indices and negative_scores by the number of positives of each query
        indices = torch.cat([indices[idx].repeat(n_positives[idx], 1) for idx in range(n_queries)])
        negative_scores = torch.cat([negative_scores[idx].repeat(n_positives[idx], 1) for idx in range(n_queries)])

        indices_to_keep = negative_scores != -float("inf")

        dataset_data = {
            anchor_column_name: [all_queries[idx] for idx, keep_row in enumerate(indices_to_keep) if keep_row.any()],
            positive_column_name: [
                [positives[idx]] + [corpus[index] for keep, index in zip(keep_row, indices_row) if keep]
                for idx, (keep_row, indices_row) in enumerate(zip(indices_to_keep, indices))
                if keep_row.any()
            ],
        }
        # Either `scores` or binary `labels` field
        if output_scores:
            dataset_data["scores"] = [
                [positive_scores[idx].item()] + negative_scores[idx][keep_row].tolist()
                for idx, keep_row in enumerate(indices_to_keep)
                if keep_row.any()
            ]
        else:
            dataset_data["labels"] = [[1] + [0] * sum(keep_row) for keep_row in indices_to_keep if keep_row.any()]

        negative_scores = negative_scores[indices_to_keep]
        difference_scores = positive_scores.repeat(num_negatives, 1).T[indices_to_keep] - negative_scores
        maximum_possible_samples = indices_to_keep.size(0)

    output_dataset = Dataset.from_dict(dataset_data)

    # Report some statistics
    if verbose:
        row_format = "{:<6} {:>14} {:>14} {:>14}"
        formatter = lambda value: (f"{value.item():.4f}" if isinstance(value, torch.Tensor) else f"{value:,}")
        print(row_format.format("Metric", "Positive", "Negative", "Difference"))
        print(
            row_format.format(
                "Count",
                formatter(len(positive_scores)),
                formatter(len(negative_scores)),
                "",
            )
        )
        for metric, function in [
            ("mean", torch.mean),
            ("median", torch.median),
            ("std", torch.std),
            ("min", lambda scores: torch.min(scores) if scores.numel() > 0 else float("inf")),
            ("25%", lambda scores: torch.quantile(scores.float(), q=0.25) if scores.numel() > 0 else float("inf")),
            ("50%", lambda scores: torch.quantile(scores.float(), q=0.5) if scores.numel() > 0 else float("inf")),
            ("75%", lambda scores: torch.quantile(scores.float(), q=0.75) if scores.numel() > 0 else float("inf")),
            ("max", lambda scores: torch.max(scores) if scores.numel() > 0 else float("-inf")),
        ]:
            print(
                row_format.format(
                    metric.capitalize(),
                    formatter(function(positive_scores)),
                    formatter(function(negative_scores)),
                    formatter(function(difference_scores)),
                )
            )

        for param_name, param_value in [
            ("absolute_margin", absolute_margin),
            ("relative_margin", relative_margin),
            ("max_score", max_score),
            ("min_score", min_score),
        ]:
            if param_name in log_counters:
                skipped = log_counters[param_name]["skipped"]
                ratio = log_counters[param_name]["ratio"]
                print(
                    f"Skipped {skipped:,} potential negatives ({ratio:.2%}) due to the {param_name} of {param_value}."
                )

        missing_samples = maximum_possible_samples - len(output_dataset)
        if missing_samples > 0:
            solutions = ["range_max"]
            if range_min > 0:
                solutions.append("range_min")
            if absolute_margin is not None:
                solutions.append("absolute_margin")
            if relative_margin is not None:
                solutions.append("relative_margin")
            if max_score is not None:
                solutions.append("max_score")
            considerations = ", ".join(solutions[:-1])
            if len(solutions) > 1:
                considerations += " and " + solutions[-1]
            missing_samples_ratio = missing_samples / maximum_possible_samples
            print(
                f"Could not find enough negatives for {missing_samples} samples ({missing_samples_ratio:.2%})."
                f" Consider adjusting the {considerations} parameter{'s' if len(solutions) > 1 else ''} if you'd like to find more valid negatives."
            )

    return output_dataset

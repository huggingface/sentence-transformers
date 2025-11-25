"""
This script evaluates embedding models truncated at different dimensions on the STS
benchmark.
"""

import argparse
import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)


# Dimension plot
def _grouped_barplot_ratios(group_name_to_x_to_y: dict[str, dict[int, float]], ax: plt.Axes | None = None) -> plt.Axes:
    # To save a pandas dependency, do from scratch in matplotlib
    if ax is None:
        ax: plt.Axes = plt.subplots()
    # Sort each by x
    group_name_to_x_to_y = {
        group_name: dict(sorted(x_to_y.items(), key=lambda x: x[0]))
        for group_name, x_to_y in group_name_to_x_to_y.items()
    }
    # Check that all x are the same
    xticks = None
    for group_name, x_to_y in group_name_to_x_to_y.items():
        _xticks = x_to_y.keys()
        if xticks is not None and _xticks != xticks:
            raise ValueError(f"{group_name} has different keys: {_xticks}")
        xticks = _xticks
    xticks = sorted(xticks)

    # Max y will be the denominator in the ratio/fraction
    group_name_to_max_y = {group_name: max(x_to_y.values()) for group_name, x_to_y in group_name_to_x_to_y.items()}
    num_groups = len(group_name_to_x_to_y)
    bar_width = np.diff(xticks).min() / (num_groups + 1)
    # bar_width is the solution to this equation:
    # Say we have the closest x1, x2 st x1 < x2, so x2 - x1 = np.diff(xticks).min().
    # (x2 - (bar_width * num_groups/2)) - (x1 + (bar_width * num_groups/2)) = bar_width
    xs = np.array(
        [
            np.linspace(
                start=xtick - ((bar_width / 2) * (num_groups - 1)),
                stop=xtick + ((bar_width / 2) * (num_groups - 1)),
                num=num_groups,
            )
            for xtick in xticks
        ]
    ).T
    # xs are the center of where the bar goes on the x axis. They have to be manually set
    min_ratio = np.inf
    for i, (group_name, x_to_y) in enumerate(group_name_to_x_to_y.items()):
        max_y = group_name_to_max_y[group_name]
        ys = [y / max_y for y in x_to_y.values()]
        min_ratio = min(min_ratio, min(ys))
        ax.bar(xs[i], ys, bar_width, label=group_name)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.grid(linestyle="--")
    ax.set_ylim(min(0.95, min_ratio), 1)
    return ax


def plot_across_dimensions(
    model_name_to_dim_to_score: dict[str, dict[int, float]],
    filename: str,
    figsize: tuple[float, float] = (7, 7),
    title: str = "STSB test score for various embedding dimensions (via truncation),\nwith and without Matryoshka loss",
) -> None:
    # Sort each by key
    model_name_to_dim_to_score = {
        model_name: dict(sorted(dim_to_score.items(), key=lambda x: x[0]))
        for model_name, dim_to_score in model_name_to_dim_to_score.items()
    }
    xticks = sorted(list(model_name_to_dim_to_score.values())[0].keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    ax1 = cast(plt.Axes, ax1)
    ax2 = cast(plt.Axes, ax2)

    # Line plot
    for model_name, dim_to_score in model_name_to_dim_to_score.items():
        ax1.plot(dim_to_score.keys(), dim_to_score.values(), label=model_name)
    ax1.set_xticks(xticks)
    ax1.set_ylabel("Spearman correlation")
    ax1.grid(linestyle="--")
    ax1.legend()

    # Bar plot
    ax2 = _grouped_barplot_ratios(model_name_to_dim_to_score, ax=ax2)
    ax2.set_xlabel("Embedding dimension")
    ax2.set_ylabel("Ratio of maximum performance")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename)


if __name__ == "__main__":
    DEFAULT_MODEL_NAMES = [
        "tomaarsen/mpnet-base-nli-matryoshka",  # fit using Matryoshka loss
        "tomaarsen/mpnet-base-nli",  # baseline
    ]
    DEFAULT_DIMENSIONS = [768, 512, 256, 128, 64]

    # Parse args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot_filename", type=str, help="Where to save the plot of results")
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=DEFAULT_MODEL_NAMES,
        help=(
            "List of models which can be loaded using "
            "sentence_transformers.SentenceTransformer(). Default: "
            f"{' '.join(DEFAULT_MODEL_NAMES)}"
        ),
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help=(
            "List of dimensions to truncate to and evaluate. Default: "
            f"{' '.join(str(dim) for dim in DEFAULT_DIMENSIONS)}"
        ),
    )

    args = parser.parse_args()
    plot_filename: str = args.plot_filename
    model_names: list[str] = args.model_names
    DIMENSIONS: list[int] = args.dimensions

    # Load STSb
    stsb_test = load_dataset("mteb/stsbenchmark-sts", split="test")
    test_evaluator = EmbeddingSimilarityEvaluator(
        stsb_test["sentence1"],
        stsb_test["sentence2"],
        [score / 5 for score in stsb_test["score"]],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-test",
    )

    # Run test_evaluator
    model_name_to_dim_to_score: dict[str, dict[int, float]] = {}
    for model_name in tqdm(model_names, desc="Evaluating models"):
        model = SentenceTransformer(model_name)
        dim_to_score: dict[int, float] = {}
        for dim in tqdm(DIMENSIONS, desc=f"Evaluating {model_name}"):
            output_path = os.path.join(model_name, f"dim-{dim}")
            os.makedirs(output_path)
            test_evaluator.truncate_dim = dim
            score = test_evaluator(model, output_path=output_path)
            print(f"Saved results to {output_path}")
            dim_to_score[dim] = score
        model_name_to_dim_to_score[model_name] = dim_to_score

    # Save plot
    plot_across_dimensions(model_name_to_dim_to_score, plot_filename)

"""Compare CPU XTR training-scoring memory and latency in isolated processes.

Prepare the baseline scorer from the unchanged base revision, then run this script in
an environment with the candidate Sentence Transformers checkout installed::

    git show BASE_REV:sentence_transformers/multi_vector_encoder/scoring/xtr.py > /tmp/xtr-baseline.py
    python examples/multi_vector_encoder/training/xtr/benchmark_scoring_memory.py \
        --baseline-source /tmp/xtr-baseline.py > xtr-results.jsonl

The baseline is executed as trusted Python source with the installed package's helpers;
those helpers must be unchanged between revisions. No models, datasets, GPU or network
access are needed. The resource-based RSS measurement requires Linux/macOS. Each sample is a cold, single scoring call in a fresh subprocess,
with one CPU thread by default. Imports and tensor creation precede timing. RSS includes
interpreter, package and input storage, not only tensors; starting high-water RSS is
reported alongside peak RSS. This is a scoring-stage benchmark, not encoder training.

Compare memory AND latency: training streaming selection can be slower, particularly
for reduced-precision cutoff ties. Forward-only samples do not require gradients and
retain legacy inference scoring. Short documents with large dimensions also exercise
the legacy fallback. There is no universal speedup or memory-budget guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

PROFILES = {
    "fp32-h64": (16, 32, 180, 64, "float32"),
    "fp32-h128": (16, 32, 180, 128, "float32"),
    "short-doc": (4, 8, 8, 1024, "float32"),
    "bf16-h128": (16, 32, 180, 128, "bfloat16"),
}


def measure(args: argparse.Namespace) -> None:
    import torch

    from sentence_transformers.multi_vector_encoder.scoring import xtr as xtr_module
    from sentence_transformers.multi_vector_encoder.scoring import xtr_scores

    baseline_source = args.baseline_source.read_bytes()
    candidate_source = Path(xtr_module.__file__).read_bytes()
    namespace = {"__file__": str(args.baseline_source)}
    exec(compile(baseline_source, str(args.baseline_source), "exec"), namespace)
    scorer = namespace["xtr_scores"] if args.mode == "baseline" else xtr_scores
    batch, ways, document_tokens, dimension, dtype_name = PROFILES[args.profile]
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(args.seed)
    dtype = getattr(torch, dtype_name)
    backward = args.pass_name == "forward-backward"
    queries = torch.randn(batch, 32, dimension, generator=generator, dtype=dtype, requires_grad=backward)
    documents = torch.randn(
        batch,
        ways,
        document_tokens,
        dimension,
        generator=generator,
        dtype=dtype,
        requires_grad=backward,
    )
    rss_divisor = 1024**2 if sys.platform == "darwin" else 1024
    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_divisor
    start = time.perf_counter()
    if args.autocast is None:
        scores = scorer(queries, documents, top_k=args.top_k, chunk_elements=args.chunk_elements)
    else:
        with torch.autocast("cpu", dtype=getattr(torch, args.autocast)):
            scores = scorer(queries, documents, top_k=args.top_k, chunk_elements=args.chunk_elements)
    if backward:
        scores.sum().backward()
    seconds = time.perf_counter() - start
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_divisor
    print(
        json.dumps(
            {
                "profile": args.profile,
                "mode": args.mode,
                "pass": args.pass_name,
                "seconds": seconds,
                "peak_rss_mib": peak_rss,
                "start_peak_rss_mib": start_rss,
                "incremental_peak_rss_mib": peak_rss - start_rss,
                "query_shape": list(queries.shape),
                "document_shape": list(documents.shape),
                "dtype": str(dtype),
                "autocast": args.autocast,
                "threads": args.threads,
                "seed": args.seed,
                "top_k": args.top_k,
                "chunk_elements": args.chunk_elements,
                "checksum": scores.sum().item(),
                "torch": torch.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "baseline_sha256": hashlib.sha256(baseline_source).hexdigest(),
                "candidate_sha256": hashlib.sha256(candidate_source).hexdigest(),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--autocast", choices=["bfloat16", "float16"], help="CPU forward autocast dtype")
    parser.add_argument(
        "--passes",
        nargs="+",
        choices=["forward", "forward-backward"],
        default=["forward", "forward-backward"],
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--chunk-elements", type=int, default=1_000_000)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", choices=PROFILES, help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=["baseline", "candidate"], help=argparse.SUPPRESS)
    parser.add_argument("--pass-name", choices=["forward", "forward-backward"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.baseline_source.is_file():
        parser.error("--baseline-source must name a trusted baseline scoring module")
    if min(args.repeats, args.threads, args.top_k, args.chunk_elements) < 1:
        parser.error("repeats, threads, top-k and chunk-elements must be positive")
    if args.worker:
        measure(args)
        return
    for profile in args.profiles:
        for repeat in range(args.repeats):
            for pass_name in args.passes:
                for mode in ("baseline", "candidate"):
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--baseline-source",
                        str(args.baseline_source.resolve()),
                        "--profile",
                        profile,
                        "--mode",
                        mode,
                        "--pass-name",
                        pass_name,
                        "--threads",
                        str(args.threads),
                        "--seed",
                        str(args.seed),
                        "--top-k",
                        str(args.top_k),
                        "--chunk-elements",
                        str(args.chunk_elements),
                    ]
                    if args.autocast is not None:
                        command.extend(["--autocast", args.autocast])
                    sample = subprocess.run(command, check=True, capture_output=True, text=True)
                    record = json.loads(sample.stdout)
                    record["repeat"] = repeat
                    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

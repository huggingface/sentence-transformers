"""
A quantized model executes some or all of the operations with integers rather than floating point values. This allows for a more compact models and the use of high performance vectorized operations on many hardware platforms.

As a result, you get about much smaller and faster models. The speed-up depends on your CPU, but you can expect a speed-up of 2x to 4x for most CPUs. The model size is also reduced by 2x.

Note: Quantized models are only recommended for CPUs. If available, Use a GPU for optimal performance.

See docs for more information on quantization, optimization, benchmarks, etc.: https://sbert.net/docs/sentence_transformer/usage/efficiency.html
"""

import logging
import time

from datasets import load_dataset

from sentence_transformers import (
    SentenceTransformer,
    export_dynamic_quantized_onnx_model,
    export_static_quantized_openvino_model,
)
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator

#### Just some code to print debug information to stdout
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# Load some sentences from the STSbenchmark dataset
train_dataset = load_dataset("sentence-transformers/stsb", split="train")
sentences = train_dataset["sentence1"] + train_dataset["sentence2"]
sentences = sentences[:10_000]
test_dataset = load_dataset("sentence-transformers/stsb", split="test")

model_name = "all-mpnet-base-v2"

# 1. Load a baseline model with just fp32 torch
model = SentenceTransformer(model_name, device="cpu")

# 2. Load an ONNX model to quantize
onnx_model = SentenceTransformer(
    model_name,
    backend="onnx",
    device="cpu",
    model_kwargs={"provider": "CPUExecutionProvider"},
)

# 3. Quantize the ONNX model
quantized_onnx_model_path = f"{model_name.replace('/', '-')}-onnx-quantized"
onnx_model.save_pretrained(quantized_onnx_model_path)
export_dynamic_quantized_onnx_model(
    onnx_model,
    quantization_config="avx512_vnni",
    model_name_or_path=quantized_onnx_model_path,
)
quantized_onnx_model = SentenceTransformer(
    quantized_onnx_model_path,
    backend="onnx",
    device="cpu",
    model_kwargs={
        "file_name": "model_qint8_avx512_vnni.onnx",
        "provider": "CPUExecutionProvider",
    },
)
# Alternatively, you can load the pre-quantized model:
# quantized_onnx_model = SentenceTransformer(
#     model_name,
#     backend="onnx",
#     device="cpu",
#     model_kwargs={
#         "file_name": "model_qint8_avx512_vnni.onnx",
#         "provider": "CPUExecutionProvider",
#     },
# )

# To make sure that `onnx_model` itself didn't get quantized, we reload it
onnx_model = SentenceTransformer(
    model_name,
    backend="onnx",
    device="cpu",
    model_kwargs={"provider": "CPUExecutionProvider"},
)

# 4. Load an OpenVINO model to quantize
openvino_model = SentenceTransformer(model_name, backend="openvino", device="cpu")

# 5. Quantize the OpenVINO model
quantized_ov_model_path = f"{model_name.replace('/', '-')}-ov-quantized"
openvino_model.save_pretrained(quantized_ov_model_path)
export_static_quantized_openvino_model(
    openvino_model,
    quantization_config=None,
    model_name_or_path=quantized_ov_model_path,
)
quantized_ov_model = SentenceTransformer(
    quantized_ov_model_path,
    backend="openvino",
    device="cpu",
    model_kwargs={"file_name": "openvino_model_qint8_quantized.xml"},
)
# Alternatively, you can load the pre-quantized model:
# quantized_ov_model = SentenceTransformer(
#     model_name,
#     backend="openvino",
#     device="cpu",
#     model_kwargs={"file_name": "openvino_model_qint8_quantized.xml"},
# )

# To make sure that `openvino_model` itself didn't get quantized, we reload it
openvino_model = SentenceTransformer(model_name, backend="openvino", device="cpu")


# Create a function to evaluate the models
def evaluate(model: SentenceTransformer, name: str) -> None:
    logging.info(f"Evaluating {name}")
    start_time = time.time()
    model.encode(sentences)
    diff = time.time() - start_time
    logging.info(f"Done after {diff:.2f} sec. {len(sentences) / diff:.2f} sentences / sec")

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=test_dataset["sentence1"],
        sentences2=test_dataset["sentence2"],
        scores=test_dataset["score"],
        name="sts-test",
    )
    results = evaluator(model)
    logging.info(f"STS Benchmark, {evaluator.primary_metric}: {results[evaluator.primary_metric]}")


# Evaluate the models
for model, name in [
    (model, "Baseline"),
    (onnx_model, "ONNX"),
    (quantized_onnx_model, "Quantized ONNX"),
    (openvino_model, "OpenVINO"),
    (quantized_ov_model, "Quantized OpenVINO"),
]:
    evaluate(model, name)

"""
Evaluating Baseline
Done after 48.79 sec. 204.97 sentences / sec
STS Benchmark, sts-test_spearman_cosine: 0.834221557992808

Evaluating ONNX
Done after 36.79 sec. 271.84 sentences / sec
STS Benchmark, sts-test_spearman_cosine: 0.8342216139244768

Evaluating Quantized ONNX
Done after 17.84 sec. 560.60 sentences / sec
STS Benchmark, sts-test_spearman_cosine: 0.8256725903061843

Evaluating OpenVINO
Done after 36.43 sec. 274.49 sentences / sec
STS Benchmark, sts-test_spearman_cosine: 0.834221557992808

Evaluating Quantized OpenVINO
Done after 12.94 sec. 772.83 sentences / sec
STS Benchmark, sts-test_spearman_cosine: 0.8315710087348848
"""

# Quantization-Aware Training (QAT)

Quantization-Aware Training (QAT) is a technique that trains models to maintain high performance even when their embeddings are quantized to lower precision formats. While standard embedding models typically use 32-bit floating-point (float32) representations, quantization allows you to reduce this to 8-bit integers (int8/uint8) or even binary formats, dramatically reducing storage costs and speeding up similarity computations.

Simply quantizing a standard model's embeddings often leads to significant performance degradation. QAT addresses this by training the model to be robust to quantization from the start.

## Why Quantization?

**Storage Efficiency**: Quantized embeddings require significantly less storage:
- **int8/uint8**: 4x smaller than float32 (8 bits vs 32 bits per dimension)
- **binary/ubinary**: 32x smaller than float32 (1 bit vs 32 bits per dimension)

**Computation Speed**: Lower precision enables faster similarity calculations, especially for large-scale retrieval tasks.

**Cost Reduction**: Smaller embeddings mean lower storage and bandwidth costs in production systems.

## Training

Training with Quantization-Aware Training is straightforward with the `QuantizationAwareLoss`. This loss modifier wraps your existing loss function and trains on multiple quantization precisions simultaneously:

```python
from sentence_transformers import SentenceTransformer, losses

model = SentenceTransformer("microsoft/mpnet-base")

# Define your base loss
base_loss = losses.MultipleNegativesRankingLoss(model=model)

# Wrap it with QuantizationAwareLoss
loss = losses.QuantizationAwareLoss(
    model=model,
    loss=base_loss,
    quantization_precisions=["float32", "int8", "binary"],
    quantization_weights=[1, 1, 1],  # Optional: weight each precision differently
)
```

The loss works by:
1. Computing embeddings in float32 (and caching them)
2. Quantizing these (cached) embeddings to each specified precision
3. Computing the loss for each precision using the quantized and cached embeddings
4. Combining all losses (with optional weighting)

### Supported Precisions

- **float32**: Standard 32-bit floating point (baseline)
- **int8**: Signed 8-bit integer quantization
- **uint8**: Unsigned 8-bit integer quantization  
- **binary**: Signed binary (1-bit) quantization
- **ubinary**: Unsigned binary (1-bit) quantization

## Examples

### Training for Retrieval

Train a model for information retrieval tasks with QAT:

```bash
python train_qat_gooaq.py
# or with a specific model
python train_qat_gooaq.py sentence-transformers/all-MiniLM-L6-v2
```

This script trains on the [GooAQ](https://huggingface.co/datasets/sentence-transformers/gooaq) dataset using `MultipleNegativesRankingLoss` wrapped in `QuantizationAwareLoss`. It's optimized for question-answering retrieval tasks and demonstrates QAT with binary quantization for maximum compression.

### Training on STS Benchmark

Train a model for semantic similarity with QAT:

```bash
python train_qat_sts.py
# or with a specific model
python train_qat_sts.py sentence-transformers/all-MiniLM-L6-v2
```

This script trains a model on the [STS Benchmark](https://huggingface.co/datasets/sentence-transformers/stsb) dataset using `CoSENTLoss` wrapped in `QuantizationAwareLoss`. The model learns to maintain performance across float32, int8, and binary precisions.

### Training on NLI

Train a model for Natural Language Inference with QAT:

```bash
python train_qat_nli.py
# or with a specific model
python train_qat_nli.py distilbert-base-uncased
```

This script trains on the [AllNLI](https://huggingface.co/datasets/sentence-transformers/all-nli) dataset using `MultipleNegativesRankingLoss` wrapped in `QuantizationAwareLoss`.

## Using Quantized Embeddings

After training, you can encode with quantization:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("path/to/your/qat-model")

# Encode with quantization
embeddings_int8 = model.encode(
    sentences,
    precision="int8",
    normalize_embeddings=True,  # Recommended for quantized embeddings
)

embeddings_binary = model.encode(
    sentences,
    precision="binary",
    normalize_embeddings=True,
)
```

You can also evaluate with quantization:

```python
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

evaluator = EmbeddingSimilarityEvaluator(
    sentences1=sentences1,
    sentences2=sentences2,
    scores=scores,
    precision="int8",  # or "binary", "uint8", etc.
)
results = evaluator(model)
```

## Related Techniques

- **[Matryoshka Embeddings](../matryoshka/README.md)**: Train models with variable output dimensions
- **[Adaptive Layer](../adaptive_layer/README.md)**: Reduce model layers for faster inference
- **Combining approaches**: You can combine QuantizationAwareLoss with MatryoshkaLoss for models that support both dimension reduction and quantization

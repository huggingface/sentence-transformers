# Embedding Quantization

Embeddings may be challenging to scale up, which leads to expensive solutions and high latencies. Currently, many state-of-the-art models produce embeddings with 1024 dimensions, each of which is encoded in `float32`, i.e., they require 4 bytes per dimension. To perform retrieval over 50 million vectors, you would therefore need around 200GB of memory. This tends to require complex and costly solutions at scale.

However, there is a new approach to counter this problem; it entails reducing the size of each of the individual values in the embedding: **Quantization**. Experiments on quantization have shown that we can maintain a large amount of performance while significantly speeding up computation and saving on memory, storage, and costs.

To learn more about Embedding Quantization and their performance, please read the [blogpost](https://huggingface.co/blog/embedding-quantization) by Sentence Transformers and mixedbread.ai.

## Binary Quantization

Binary quantization refers to the conversion of the `float32` values in an embedding to 1-bit values, resulting in a 32x reduction in memory and storage usage. To quantize `float32` embeddings to binary, we simply threshold normalized embeddings at 0: if the value is larger than 0, we make it 1, otherwise we convert it to 0. We can use the Hamming Distance to efficiently perform retrieval with these binary embeddings. This is simply the number of positions at which the bits of two binary embeddings differ. The lower the Hamming Distance, the closer the embeddings, and thus the more relevant the document. A huge advantage of the Hamming Distance is that it can be easily calculated with 2 CPU cycles, allowing for blazingly fast performance.

[Yamada et al. (2021)](https://huggingface.co/papers/2106.00882) introduced a rescore step, which they called *rerank*, to boost the performance. They proposed that the `float32` query embedding could be compared with the binary document embeddings using dot-product. In practice, we first retrieve `rescore_multiplier * top_k` results with the binary query embedding and the binary document embeddings -- i.e., the list of the first k results of the double-binary retrieval -- and then rescore that list of binary document embeddings with the `float32` query embedding.

By applying this novel rescoring step, we are able to preserve up to ~96% of the total retrieval performance, while reducing the memory and disk space usage by 32x and improving the retrieval speed by up to 32x as well.

### Binary Quantization in Sentence Transformers

Quantizing an embedding with a dimensionality of 1024 to binary would result in 1024 bits. In practice, it is much more common to store bits as bytes instead, so when we quantize to binary embeddings, we pack the bits into bytes using `np.packbits`.

As a result, in practice quantizing a `float32` embedding with a dimensionality of 1024 yields an `int8` or `uint8` embedding with a dimensionality of 128. See two approaches of how you can produce quantized embeddings using Sentence Transformers below:

```{eval-rst}
.. sidebar:: References

   #. `mixedbread-ai/mxbai-embed-large-v1 <https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1>`_
   #. :class:`~sentence_transformers.SentenceTransformer`
   #. :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>`
   #. :func:`~sentence_transformers.quantization.quantize_embeddings`

```

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

# 1. Load an embedding model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# 2a. Encode some text using "binary" quantization
binary_embeddings = model.encode(
    ["I am driving to the lake.", "It is a beautiful day."],
    precision="binary",
)

# 2b. or, encode some text without quantization & apply quantization afterwards
embeddings = model.encode(["I am driving to the lake.", "It is a beautiful day."])
binary_embeddings = quantize_embeddings(embeddings, precision="binary")
```

Here you can see the differences between default `float32` embeddings and binary embeddings in terms of shape, size, and `numpy` dtype:

```python
>>> embeddings.shape
(2, 1024)
>>> embeddings.nbytes
8192
>>> embeddings.dtype
float32
>>> binary_embeddings.shape
(2, 128)
>>> binary_embeddings.nbytes
256
>>> binary_embeddings.dtype
int8
```

Note that you can also choose `"ubinary"` to quantize to binary using the unsigned `uint8` data format. This may be a requirement for your vector library/database.

## Scalar (int8) Quantization

To convert the `float32` embeddings into `int8`, we use a process called scalar quantization. This involves mapping the continuous range of `float32` values to the discrete set of `int8` values, which can represent 256 distinct levels (from -128 to 127). This is done by using a large calibration dataset of embeddings. We compute the range of these embeddings, i.e. the `min` and `max` of each of the embedding dimensions. From there, we calculate the steps (buckets) in which we categorize each value.

To further boost the retrieval performance, you can optionally apply the same rescoring step as for the binary embeddings. It is important to note here that the calibration dataset has a large influence on the performance, since it defines the buckets.

### Scalar Quantization in Sentence Transformers

Quantizing an embedding with a dimensionality of 1024 to `int8` results in 1024 bytes. In practice, we can choose either `uint8` or `int8`. This choice is usually made depending on what your vector library/database supports.

In practice, it is recommended to provide the scalar quantization with either:

1. a large set of embeddings to quantize all at once, or
1. `min` and `max` ranges for each of the embedding dimensions, or
1. a large calibration dataset of embeddings from which the `min` and `max` ranges can be computed.

If none of these are the case, you will be given a warning like this:

```
Computing int8 quantization buckets based on 2 embeddings. int8 quantization is more stable with 'ranges' calculated from more embeddings or a 'calibration_embeddings' that can be used to calculate the buckets.
```

See how you can produce scalar quantized embeddings using Sentence Transformers below:

```{eval-rst}
.. sidebar:: References

   #. `mixedbread-ai/mxbai-embed-large-v1 <https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1>`_
   #. :class:`~sentence_transformers.SentenceTransformer`
   #. :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>`
   #. :func:`~sentence_transformers.quantization.quantize_embeddings`

```

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from datasets import load_dataset

# 1. Load an embedding model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# 2. Prepare an example calibration dataset
corpus = load_dataset("nq_open", split="train[:1000]")["question"]
calibration_embeddings = model.encode(corpus)

# 3. Encode some text without quantization & apply quantization afterwards
embeddings = model.encode(["I am driving to the lake.", "It is a beautiful day."])
int8_embeddings = quantize_embeddings(
    embeddings,
    precision="int8",
    calibration_embeddings=calibration_embeddings,
)
```

Here you can see the differences between default `float32` embeddings and `int8` scalar embeddings in terms of shape, size, and `numpy` dtype:

```python
>>> embeddings.shape
(2, 1024)
>>> embeddings.nbytes
8192
>>> embeddings.dtype
float32
>>> int8_embeddings.shape
(2, 1024)
>>> int8_embeddings.nbytes
2048
>>> int8_embeddings.dtype
int8
```

### Combining Binary and Scalar Quantization

It is possible to combine binary and scalar quantization to get the best of both worlds: the extreme speed from binary embeddings and the great performance preservation of scalar embeddings with rescoring. See the [demo](#demo) below for a real-life implementation of this approach involving 41 million texts from Wikipedia. The pipeline for that setup is as follows:

1. The query is embedded using the [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) SentenceTransformer model.
1. The query is quantized to binary using the <a href="../../../../docs/package_reference/quantization.html#sentence_transformers.quantization.quantize_embeddings"><code>quantize_embeddings</code></a> function from the `sentence-transformers` library.
1. A binary index (41M binary embeddings; 5.2GB of memory/disk space) is searched using the quantized query for the top 40 documents.
1. The top 40 documents are loaded on the fly from an int8 index on disk (41M int8 embeddings; 0 bytes of memory, 47.5GB of disk space).
1. The top 40 documents are rescored using the float32 query and the int8 embeddings to get the top 10 documents.
1. The top 10 documents are sorted by score and displayed.

Through this approach, we use 5.2GB of memory and 52GB of disk space for the indices. This is considerably less than normal retrieval, for which we would require 200GB of memory and 200GB of disk space. Especially as you scale up even further, this will result in notable reductions in both latency and costs.

## Additional extensions

Note that embedding quantization can be combined with other approaches to improve retrieval efficiency, such as [Matryoshka Embeddings](../../training/matryoshka/README.md). Additionally, the [Retrieve & Re-Rank](../retrieve_rerank/README.md) also works very well with quantized embeddings, i.e. you can still use a Cross-Encoder to rerank.

## Demo

The following demo showcases the retrieval efficiency using `exact` search through combining binary search with scalar (`int8`) rescoring. The solution requires 5GB of memory for the binary index and 50GB of disk space for the binary and scalar indices, considerably less than the 200GB of memory and disk space which would be required for regular `float32` retrieval. Additionally, retrieval is much faster.

<iframe
	src="https://sentence-transformers-quantized-retrieval.hf.space"
	frameborder="0"
	width="100%"
	height="1000"
></iframe>
<p></p> <!-- Force a newline -->

## Try it yourself

The following scripts can be used to experiment with embedding quantization for retrieval & beyond. There are three categories:

- **Recommended Retrieval**:
  - [semantic_search_recommended.py](semantic_search_recommended.py): This script combines binary search with scalar rescoring, much like the above demo, for cheap, efficient, and performant retrieval.
- **Usage**:
  - [semantic_search_faiss.py](semantic_search_faiss.py): This script showcases regular usage of binary or scalar quantization, retrieval, and rescoring using FAISS, by using the <a href="../../../../docs/package_reference/quantization.html#sentence_transformers.quantization.semantic_search_faiss"><code>semantic_search_faiss</code></a> utility function.
  - [semantic_search_usearch.py](semantic_search_usearch.py): This script showcases regular usage of binary or scalar quantization, retrieval, and rescoring using USearch, by using the <a href="../../../../docs/package_reference/quantization.html#sentence_transformers.quantization.semantic_search_usearch"><code>semantic_search_usearch</code></a> utility function.
- **Benchmarks**:
  - [semantic_search_faiss_benchmark.py](semantic_search_faiss_benchmark.py): This script includes a retrieval speed benchmark of `float32` retrieval, binary retrieval + rescoring, and scalar retrieval + rescoring, using FAISS. It uses the <a href="../../../../docs/package_reference/quantization.html#sentence_transformers.quantization.semantic_search_faiss"><code>semantic_search_faiss</code></a> utility function. Our benchmarks especially show show speedups for `ubinary`.
  - [semantic_search_usearch_benchmark.py](semantic_search_usearch_benchmark.py): This script includes a retrieval speed benchmark of `float32` retrieval, binary retrieval + rescoring, and scalar retrieval + rescoring, using USearch. It uses the <a href="../../../../docs/package_reference/quantization.html#sentence_transformers.quantization.semantic_search_usearch"><code>semantic_search_usearch</code></a> utility function. Our experiments show large speedups on newer hardware, particularly for `int8`.

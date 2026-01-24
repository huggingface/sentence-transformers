# Fine-tune Embedding Models with Unsloth 

[Unsloth](https://unsloth.ai/docs) is an open-source training framework that also supports fine-tuning embedding, classifier, BERT, reranker models [~1.8-3.3x faster](##unsloth-benchmarks) with 20% less memory and 2x longer context than setups with Flash Attention 2 with no accuracy degradation. Visit the [Unsloth GitHub repo](https://github.com/unslothai/unsloth).

Unsloth uses Hugging Face `sentence-transformers` to support most SentenceTransformers compatible models like Qwen3-Embedding, BERT, and more.

  
Models like EmbeddingGemma-300M work on 3GB VRAM. You can use your trained model anywhere: sentencetransformers, transformers, LangChain, Ollama, vLLM, llama.cpp etc. 

Unsloth has multiple free fine-tuning notebooks, with many use-cases:
- [EmbeddingGemma (300M)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma_(300M).ipynb)
- [Qwen3-Embedding 4B](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_Embedding_(4B).ipynb) • [0.6B](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_Embedding_(0_6B).ipynb)
- [BGE M3](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/BGE_M3.ipynb)
- [ModernBERT-large](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/bert_classification.ipynb)
- [All-MiniLM-L6-v2](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/All_MiniLM_L6_v2.ipynb)
- [GTE ModernBert](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/ModernBert.ipynb)

You can easily change the model name in the config to use another model. Below are the datasets and use-cases the notebooks contain:
* `All-MiniLM-L6-v2`: produce compact, domain-specific sentence embeddings for semantic search, retrieval, and clustering, tuned on your own data.
* `tomaarsen/miriad-4.4M-split`: embed medical questions and biomedical papers for high-quality medical semantic search and RAG.
* `electroglyph/technical`: better capture meaning and semantic similarity in technical text (docs, specs, and engineering discussions).

You can view the rest of Unsloth's uploaded models in [our collection here](https://huggingface.co/collections/unsloth/embedding-models).

## 🦥 Unsloth Features

* LoRA/QLoRA or full fine-tuning for embeddings, without needing to rewrite your pipeline
* Best support for encoder-only `SentenceTransformer` models (with a `modules.json`)
* Cross-encoder models are confirmed to train properly even under the fallback path
* This release also supports `transformers v5`

There is limited support for models without `modules.json` (we’ll auto-assign default `SentenceTransformers` pooling modules). If you’re doing something custom (custom heads, nonstandard pooling), double-check outputs like the pooled embedding behavior.

Some models needed custom additions such as MPNet or DistilBERT were enabled by patching gradient checkpointing into the `transformers` models.

## 🛠️ Fine-tuning Workflow

The new fine-tuning flow is centered around `FastSentenceTransformer`.

Main save/push methods:

* `save_pretrained()` Saves LoRA adapters to a local folder
* `save_pretrained_merged()` Saves the merged model to a local folder
* `push_to_hub()` Pushes LoRA adapters to Hugging Face
* `push_to_hub_merged()` Pushes the merged model to Hugging Face

And one very important detail: Inference loading requires `for_inference=True`

`from_pretrained()` is similar to Lacker’s other fast classes, with one exception:

* To load a model for inference using `FastSentenceTransformer`, you must pass: `for_inference=True`

So your inference loads should look like:

    model = FastSentenceTransformer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        for_inference=True,
    )

For Hugging Face authorization, if you run:

    hf auth login

inside the same virtualenv before calling the hub methods, then:

* `push_to_hub()` and `push_to_hub_merged()` don’t require a token argument.

## ✅ Inference and Deploy Anywhere!

Your fine-tuned Unsloth model can be used and deployed with all major tools: sentence-transformers, transformers, LangChain, Weaviate, Text Embeddings Inference (TEI), vLLM, and llama.cpp, custom embedding API, pgvector, FAISS/vector databases, and any RAG framework.

There is no lock in as the fine-tuned model can later be downloaded locally on your own device.

    # 1. Load a pretrained Sentence Transformer model
    model = SentenceTransformer("<your-unsloth-finetuned-model")
    
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
    ]
    
    # 2. Encode via encode_query and encode_document to automatically use the right prompts, if needed
    query_embedding = model.encode_query(query)
    document_embedding = model.encode_document(documents)
    print(query_embedding.shape, document_embedding.shape)
    
    # 3. Compute similarity, e.g. via the built-in similarity helper function
    similarity = model.similarity(query_embedding, document_embedding)
    print(similarity)

## 📊 Unsloth Benchmarks

Unsloth's advantages include speed for embedding fine-tuning! Unsloth benchmarks showcase that they are 1.8 to 3.3x faster on a wide variety of embedding models and on different sequence lengths from 128 to 2048 and longer.

EmbeddingGemma-300M QLoRA works on just 3GB VRAM and LoRA works on 6GB VRAM.

For visualizations of Unsloth benchmarks in a heatmap vs. optimized setups with Flash Attention 2 (FA2), you can [visit their docs](https://unsloth.ai/docs/new/embedding-finetuning#unsloth-benchmarks). For 4bit QLoRA, Unsloth is 1.8x to 2.6x faster and for 16bit LoRA. For 16bit LoRA, Unsloth is 1.2x to 3.3x faster, depending on the model and dataset.


## 🔮 Model Support

Here are some popular embedding models Unsloth supports (not all models are listed here):

    Alibaba-NLP/gte-modernbert-base
    BAAI/bge-large-en-v1.5
    BAAI/bge-m3
    BAAI/bge-reranker-v2-m3
    Qwen/Qwen3-Embedding-0.6B
    answerdotai/ModernBERT-base
    answerdotai/ModernBERT-large
    google/embeddinggemma-300m
    intfloat/e5-large-v2
    intfloat/multilingual-e5-large-instruct
    mixedbread-ai/mxbai-embed-large-v1
    sentence-transformers/all-MiniLM-L6-v2
    sentence-transformers/all-mpnet-base-v2
    Snowflake/snowflake-arctic-embed-l-v2.0

Most [common models](https://huggingface.co/models?library=sentence-transformers) are already supported.

  
### Resources

To get more information on Unsloth, you can visit their [GitHub repo](https://github.com/unslothai/unsloth) or their [fine-tuning embedding models article](https://unsloth.ai/docs/new/embedding-finetuning).

# Image Search

SentenceTransformers provides models that allow to embed images and text into the same vector space. This allows to find similar images as well as to implement **image search**.

![ImageSearch](https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/ImageSearch.png)

## Installation

Ensure that you have [transformers](https://pypi.org/project/transformers/) installed to use the image-text-models and use a recent PyTorch version (tested with PyTorch 1.7.0). Image-Text-Models have been added with SentenceTransformers version 1.0.0. Image-Text-Models are still in an experimental phase.

## Usage

SentenceTransformers provides a wrapper for the [OpenAI CLIP Model](https://github.com/openai/CLIP), which was trained on a variety of (image, text)-pairs.

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

# Load CLIP model
model = SentenceTransformer("clip-ViT-B-32")

# Encode an image:
img_emb = model.encode(Image.open("two_dogs_in_snow.jpg"))

# Encode text descriptions
text_emb = model.encode(
    ["Two dogs in the snow", "A cat on a table", "A picture of London at night"]
)

# Compute similarities
similarity_scores = model.similarity(img_emb, text_emb)
print(similarity_scores)
```

You can use the CLIP model for:

- Text-to-Image / Image-To-Text / Image-to-Image / Text-to-Text Search
- You can fine-tune it on your own image&text data with the regular SentenceTransformers training code.

## Examples

- [Image_Search.ipynb](Image_Search.ipynb) ([Colab Version](https://colab.research.google.com/drive/16OdADinjAg3w3ceZy3-cOR9A-5ZW9BYr?usp=sharing)) depicts a larger example for **text-to-image** and **image-to-image** search using 25,000 free pictures from [Unsplash](https://unsplash.com/).
- [Image_Search-multilingual.ipynb](Image_Search-multilingual.ipynb) ([Colab Version](https://colab.research.google.com/drive/1N6woBKL4dzYsHboDNqtv-8gjZglKOZcn?usp=sharing)) example of multilingual text2image search for 50+ languages.
- [Image_Clustering.ipynb](Image_Clustering.ipynb) ([Colab Version](https://colab.research.google.com/drive/1T3gfEF7pkXgPPajNa9ZjurB25B0RJ3_X?usp=sharing)) shows how to perform **image clustering**. Given 25,000 free pictures from [Unsplash](https://unsplash.com/), we find clusters of similar images. You can control how sensitive the clustering should be.
- [Image_Duplicates.ipynb](Image_Duplicates.ipynb) ([Colab Version](https://colab.research.google.com/drive/1wLiZNedMwlM-FxBVbp3aA353yohV_wJ1?usp=sharing)) shows an example how to find duplicate and near duplicate images in a large collection of photos.
- [Image_Classification.ipynb](Image_Classification.ipynb) ([Colab Version](https://colab.research.google.com/drive/1J0a29kSZ7qJwu2bGqjo1GYRz77v1oWq0?usp=sharing)) example for (multi-lingual) zero-shot image classifcation.

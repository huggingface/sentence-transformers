# Installation

We recommend **Python 3.10+**, **[PyTorch 1.11.0+](https://pytorch.org/get-started/locally/)**, and **[transformers v4.41.0+](https://github.com/huggingface/transformers)**. There are several extra options to install Sentence Transformers:

- **Default:** Allows loading, saving, and inference (i.e., getting embeddings) of text models.
- **Image:** Adds dependencies for models that process images (e.g., CLIP, VLM-based models).
- **Audio:** Adds dependencies for models that process audio inputs.
- **Video:** Adds dependencies for models that process video inputs.
- **Training:** Adds dependencies for training and finetuning models.
- **ONNX:** Adds dependencies for loading, saving, inference, optimizing, and quantizing of models using the ONNX backend.
- **OpenVINO:** Adds dependencies for loading, saving, and inference of models using the OpenVINO backend.
- **Development**: All of the above plus some dependencies for developing Sentence Transformers, see [Editable Install](#editable-install).

```{eval-rst}
| To pass :class:`torchcodec.AudioDecoder <torchcodec.decoders.AudioDecoder>` or :class:`torchcodec.VideoDecoder <torchcodec.decoders.VideoDecoder>` instances as inputs, you must install `torchcodec <https://github.com/pytorch/torchcodec>`_ separately, e.g. with: ``pip install torchcodec``.
| Note that you can mix and match the various extras, e.g. ``pip install -U "sentence-transformers[train,image,video,onnx-gpu]"``.
```

## Install with uv

```{eval-rst}

.. tab:: Default

    ::

        uv pip install -U sentence-transformers

.. tab:: Image

    ::

        uv pip install -U "sentence-transformers[image]"

.. tab:: Audio

    ::

        uv pip install -U "sentence-transformers[audio]"

.. tab:: Video

    ::

        uv pip install -U "sentence-transformers[video]"

.. tab:: Training

    ::

        uv pip install -U "sentence-transformers[train]"

    To use `Weights and Biases <https://wandb.ai/>`_ or `Trackio <https://github.com/gradio-app/trackio>`_ to track your training logs, you should also install ``wandb`` or ``trackio`` **(recommended)**::

        uv pip install trackio
    
    And to track your carbon emissions while training and have this information automatically included in your model cards, also install ``codecarbon`` **(recommended)**::

        uv pip install codecarbon

    Don't forget to add the module names to ``report_to`` in the Training Arguments when training, or they will not be used.

.. tab:: ONNX

    For GPU and CPU:
    ::

        uv pip install -U "sentence-transformers[onnx-gpu]"

    For CPU only:
    ::

        uv pip install -U "sentence-transformers[onnx]"

.. tab:: OpenVINO

    ::

        uv pip install -U "sentence-transformers[openvino]"

.. tab:: Development

    ::

        uv pip install -U "sentence-transformers[dev]"

```

## Install with pip

```{eval-rst}

.. tab:: Default

    ::

        pip install -U sentence-transformers

.. tab:: Image

    ::

        pip install -U "sentence-transformers[image]"

.. tab:: Audio

    ::

        pip install -U "sentence-transformers[audio]"

.. tab:: Video

    ::

        pip install -U "sentence-transformers[video]"

.. tab:: Training

    ::

        pip install -U "sentence-transformers[train]"

    To use `Weights and Biases <https://wandb.ai/>`_ or `Trackio <https://github.com/gradio-app/trackio>`_ to track your training logs, you should also install ``wandb`` or ``trackio`` **(recommended)**::

        pip install trackio
    
    And to track your carbon emissions while training and have this information automatically included in your model cards, also install ``codecarbon`` **(recommended)**::

        pip install codecarbon

    Don't forget to add the module names to ``report_to`` in the Training Arguments when training, or they will not be used.

.. tab:: ONNX

    For GPU and CPU:
    ::

        pip install -U "sentence-transformers[onnx-gpu]"

    For CPU only:
    ::

        pip install -U "sentence-transformers[onnx]"

.. tab:: OpenVINO

    ::

        pip install -U "sentence-transformers[openvino]"

.. tab:: Development

    ::

        pip install -U "sentence-transformers[dev]"

```

## Install with Conda

The base package is available on conda-forge. Extras (e.g. `[image]`, `[train]`) are a pip concept and not available via conda, so they are installed with pip.

```{eval-rst}

.. tab:: Default

    ::

        conda install -c conda-forge sentence-transformers

.. tab:: Image

    ::

        pip install -U "sentence-transformers[image]"

.. tab:: Audio

    ::

        pip install -U "sentence-transformers[audio]"

.. tab:: Video

    ::

        pip install -U "sentence-transformers[video]"

.. tab:: Training

    ::

        conda install -c conda-forge sentence-transformers accelerate datasets

    To use `Weights and Biases <https://wandb.ai/>`_ or `Trackio <https://github.com/gradio-app/trackio>`_ to track your training logs, you should also install ``wandb`` or ``trackio`` **(recommended)**::

        pip install trackio
    
    And to track your carbon emissions while training and have this information automatically included in your model cards, also install ``codecarbon`` **(recommended)**::

        pip install codecarbon

    Don't forget to add the module names to ``report_to`` in the Training Arguments when training, or they will not be used.

.. tab:: ONNX

    For GPU and CPU:
    ::

        pip install -U "sentence-transformers[onnx-gpu]"

    For CPU only:
    ::

        pip install -U "sentence-transformers[onnx]"

.. tab:: OpenVINO

    ::

        pip install -U "sentence-transformers[openvino]"

.. tab:: Development

    ::

        conda install -c conda-forge sentence-transformers accelerate datasets pre-commit pytest ruff

```

## Install from Source

You can install `sentence-transformers` directly from source to take advantage of the bleeding edge `main` branch rather than the latest stable release:

```{eval-rst}

.. tab:: Default

    ::

        pip install git+https://github.com/huggingface/sentence-transformers.git

.. tab:: Image

    ::

        pip install -U "sentence-transformers[image] @ git+https://github.com/huggingface/sentence-transformers.git"

.. tab:: Audio

    ::

        pip install -U "sentence-transformers[audio] @ git+https://github.com/huggingface/sentence-transformers.git"

.. tab:: Video

    ::

        pip install -U "sentence-transformers[video] @ git+https://github.com/huggingface/sentence-transformers.git"

.. tab:: Training

    ::

        pip install -U "sentence-transformers[train] @ git+https://github.com/huggingface/sentence-transformers.git"

    To use `Weights and Biases <https://wandb.ai/>`_ or `Trackio <https://github.com/gradio-app/trackio>`_ to track your training logs, you should also install ``wandb`` or ``trackio`` **(recommended)**::

        pip install trackio
    
    And to track your carbon emissions while training and have this information automatically included in your model cards, also install ``codecarbon`` **(recommended)**::

        pip install codecarbon

    Don't forget to add the module names to ``report_to`` in the Training Arguments when training, or they will not be used.

.. tab:: ONNX

    For GPU and CPU:
    ::

        pip install -U "sentence-transformers[onnx-gpu] @ git+https://github.com/huggingface/sentence-transformers.git"

    For CPU only:
    ::

        pip install -U "sentence-transformers[onnx] @ git+https://github.com/huggingface/sentence-transformers.git"

.. tab:: OpenVINO

    ::

        pip install -U "sentence-transformers[openvino] @ git+https://github.com/huggingface/sentence-transformers.git"

.. tab:: Development

    ::

        pip install -U "sentence-transformers[dev] @ git+https://github.com/huggingface/sentence-transformers.git"

```

## Editable Install

If you want to make changes to `sentence-transformers`, you will need an editable install. Clone the repository and install it with these commands:

```
git clone https://github.com/huggingface/sentence-transformers
cd sentence-transformers
pip install -e ".[train,dev]"
```

These commands will link the new `sentence-transformers` folder and your Python library paths, such that this folder will be used when importing `sentence-transformers`.

## Install PyTorch with CUDA support

To use a GPU/CUDA, you must install PyTorch with CUDA support. Follow [PyTorch - Get Started](https://pytorch.org/get-started/locally/) for installation steps.

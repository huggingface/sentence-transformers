# Installation

We recommend **Python 3.10+**, **[PyTorch 1.11.0+](https://pytorch.org/get-started/locally/)**, and **[transformers v4.41.0+](https://github.com/huggingface/transformers)**. There are 5 extra options to install Sentence Transformers:

- **Default:** This allows for loading, saving, and inference (i.e., getting embeddings) of models.
- **ONNX:** This allows for loading, saving, inference, optimizing, and quantizing of models using the ONNX backend.
- **OpenVINO:** This allows for loading, saving, and inference of models using the OpenVINO backend.
- **Default and Training**: Like **Default**, plus training.
- **Development**: All of the above plus some dependencies for developing Sentence Transformers, see [Editable Install](#editable-install).

Note that you can mix and match the various extras, e.g. `pip install -U "sentence-transformers[train,onnx-gpu]"`.

## Install with pip

```{eval-rst}

.. tab:: Default

    ::

        pip install -U sentence-transformers

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

.. tab:: Default and Training

    ::

        pip install -U "sentence-transformers[train]"

    To use `Weights and Biases <https://wandb.ai/>`_ to track your training logs, you should also install ``wandb`` **(recommended)**::

        pip install wandb
    
    And to track your Carbon Emissions while training and have this information automatically included in your model cards, also install ``codecarbon`` **(recommended)**::

        pip install codecarbon

.. tab:: Development

    ::

        pip install -U "sentence-transformers[dev]"

```

## Install with Conda

```{eval-rst}

.. tab:: Default

    ::

        conda install -c conda-forge sentence-transformers

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

.. tab:: Default and Training

    ::

        conda install -c conda-forge sentence-transformers accelerate datasets

    To use `Weights and Biases <https://wandb.ai/>`_ to track your training logs, you should also install ``wandb`` **(recommended)**::

        pip install wandb
    
    And to track your Carbon Emissions while training and have this information automatically included in your model cards, also install ``codecarbon`` **(recommended)**::

        pip install codecarbon

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

.. tab:: Default and Training

    ::

        pip install -U "sentence-transformers[train] @ git+https://github.com/huggingface/sentence-transformers.git"

    To use `Weights and Biases <https://wandb.ai/>`_ to track your training logs, you should also install ``wandb`` **(recommended)**::

        pip install wandb
    
    And to track your carbon emissions while training and have this information automatically included in your model cards, also install ``codecarbon`` **(recommended)**::

        pip install codecarbon

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

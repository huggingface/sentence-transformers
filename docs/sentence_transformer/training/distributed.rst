
Distributed Training
====================

Sentence Transformers implements two forms of distributed training: Data Parallel (DP) and Distributed Data Parallel (DDP). Read the `Data Parallelism documentation <https://huggingface.co/docs/transformers/en/perf_train_gpu_many#data-parallelism>`_ on Hugging Face for more details on these strategies. Some of the key differences include:

1. DDP is generally faster than DP because it has to communicate less data.
2. With DP, GPU 0 does the bulk of the work, while with DDP, the work is distributed more evenly across all GPUs.
3. DDP allows for training across multiple machines, while DP is limited to a single machine.

In short, **DDP is generally recommended**. You can use DDP by running your normal training scripts with ``torchrun`` or ``accelerate``. For example, if you have a script called ``train_script.py``, you can run it with DDP using the following command:

.. |br| raw:: html

   <div style="line-height: 0; padding: 0; margin: 0"></div>

.. tab:: Via ``torchrun``

   |br|

   - `torchrun documentation <https://pytorch.org/docs/stable/elastic/run.html>`_

   ::

      torchrun --nproc_per_node=4 train_script.py
   
.. tab:: Via ``accelerate``

   |br|

   - `accelerate documentation <https://huggingface.co/docs/accelerate/en/index>`_

   ::
      
      accelerate launch --num_processes 4 train_script.py

.. note::
  
   When performing distributed training, you have to wrap your code in a ``main`` function and call it with ``if __name__ == "__main__":``. This is because each process will run the entire script, so you don't want to run the same code multiple times. Here is an example of how to do this::

      from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
      # Other imports here

      def main():
          # Your training code here

      if __name__ == "__main__":
          main()

.. note::

   When using an `Evaluator <../training_overview.html#evaluator>`_, the evaluator only runs on the first device unlike the training and evaluation datasets, which are shared across all devices. 

Comparison
----------

The following table shows the speedup of DDP over DP and no parallelism given a certain hardware setup.

- Hardware: a ``p3.8xlarge`` AWS instance, i.e. 4x V100 GPUs
- Model being trained: `microsoft/mpnet-base <https://huggingface.co/microsoft/mpnet-base>`_ (133M parameters)
- Maximum sequence length: 384 (following `all-mpnet-base-v2 <https://huggingface.co/sentence-transformers/all-mpnet-base-v2>`_)
- Training datasets: MultiNLI, SNLI and STSB (note: these have short texts)
- Losses: :class:`~sentence_transformers.sentence_transformer.losses.SoftmaxLoss` for MultiNLI and SNLI, :class:`~sentence_transformers.sentence_transformer.losses.CosineSimilarityLoss` for STSB
- Batch size per device: 32

.. list-table::
   :header-rows: 1

   * - Strategy
     - Launcher
     - Samples per Second
   * - No Parallelism
     - ``CUDA_VISIBLE_DEVICES=0 python train_script.py``
     - 2724
   * - Data Parallel (DP)
     - ``python train_script.py`` (DP is used by default when launching a script with ``python``)
     - 3675 (1.349x speedup)
   * - **Distributed Data Parallel (DDP)**
     - ``torchrun --nproc_per_node=4 train_script.py`` or ``accelerate launch --num_processes 4 train_script.py``
     - **6980 (2.562x speedup)**

FSDP
----

Fully Sharded Data Parallelism (FSDP) is another distributed training strategy that is not fully supported by Sentence Transformers. It is a more advanced version of DDP that is particularly useful for very large models. Note that in the previous comparison, FSDP reaches 5782 samples per second (2.122x speedup), i.e. **worse than DDP**. FSDP only makes sense with very large models. If you want to use FSDP with Sentence Transformers, you have to be aware of the following limitations:

- You can't use the ``evaluator`` functionality with FSDP.
- You have to save the trained model with ``trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")`` followed with ``trainer.save_model("output")``.
- You have to use ``fsdp=["full_shard", "auto_wrap"]`` and ``fsdp_config={"transformer_layer_cls_to_wrap": "BertLayer"}`` in your ``SentenceTransformerTrainingArguments``, where ``BertLayer`` is the repeated layer in the encoder that houses the multi-head attention and feed-forward layers, so e.g. ``BertLayer`` or ``MPNetLayer``.

Read the `FSDP documentation <https://huggingface.co/docs/accelerate/en/usage_guides/fsdp>`_ by Accelerate for more details.
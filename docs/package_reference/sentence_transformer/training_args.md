# Training Arguments

## SentenceTransformerTrainingArguments

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.training_args.SentenceTransformerTrainingArguments
    :members:
    :inherited-members:
```

### Synchronized task-dependent training batches

`SentenceTransformerTrainingArguments.synchronized_dataset_batch_sizes` opts in to
same-task global microsteps with a **per-rank** batch size for every training task:

```python
from sentence_transformers import SentenceTransformerTrainingArguments

args = SentenceTransformerTrainingArguments(
    output_dir="output",
    synchronized_dataset_batch_sizes={"retrieval": 8, "similarity": 2},
    multi_dataset_batch_sampler="round_robin",
    dataloader_drop_last=True,
    dataloader_num_workers=0,
    accelerator_config={"split_batches": False, "dispatch_batches": False},
)
```

Pass these arguments to `SentenceTransformerTrainer` with a training `DatasetDict`
and loss mapping whose keys are exactly `"retrieval"` and `"similarity"`. With two
ranks, the example produces 16 retrieval examples or 4 similarity examples per
global microstep, with every rank processing the same task. These sizes override
the nominal `per_device_train_batch_size` for training only; evaluation is unchanged.

The default is `None`, preserving the existing sampling path. This opt-in requires
Accelerate's nullable variable-batch metadata support
([Accelerate PR #4491](https://github.com/huggingface/accelerate/pull/4491), not yet
released). Until that dependency is available, this feature is not usable with an
unpatched release. Incompatible versions are explicitly rejected when preparing
the loader; the default dependency minimum has not been raised to a nonexistent
release. It supports plain map-style
`DatasetDict` inputs, ordinary `BATCH_SAMPLER`, fixed-world-size data parallelism,
and complete gradient-accumulation windows. Streaming, custom sampler overrides,
`GROUP_BY_LABEL`, `NO_DUPLICATES` (including hashed), DataParallel, split/dispatch
batches, stateful loaders, automatic batch-size search and `ignore_data_skip=True`
are unsupported. Global text deduplication is **not** provided.

For task size `N_d`, per-rank batch size `b_d` and world size `W`, the available
number of global batches is `c_d = floor(N_d / (W * b_d))`, which must be positive.
Proportional sampling retains `q_d = c_d`; round robin retains
`q_d = min(c_j)` for every task, in DatasetDict order. The actual task frequency is
`q_d / sum(q_j)` and retained row count is `q_d * W * b_d`. All remaining rows are
dropped, including longer tasks' surplus full batches under round robin. No rows
are padded, repeated or implicitly resampled. The logged and saved plan exposes
these counts. `preserve_dataset_batch_frequencies=True` additionally rejects any
task whose original full local batch count `floor(N_d / b_d)` is not divisible by
`W`; it does not preserve legacy sample order or training trajectories.

The prepared loader has exactly `T = sum(q_d)` microsteps per epoch and requires
`T % gradient_accumulation_steps == 0`. Each global microstep emits `W` consecutive
same-task rank slots; Accelerate performs the only rank selection. Epochs use
deterministic independent sampling streams. Loss mathematics are unchanged:
CoSENT retains its rank-local objective averaged by DDP, not global concatenated
CoSENT. `CachedMultipleNegativesRankingLoss(gather_across_devices=True)` retains
its native global-query mean objective. For an accumulation window of `G`
microsteps, the objective is the mean of those `G` microstep objectives, not a
loss computed on their pooled examples. This mode does not add cross-rank label
comparisons to CoSENT or change either loss's scaling.

Checkpoint manifests reject changes to task order, data fingerprints/lengths,
batch sizes, world size, strategy, seed, accumulation and retained schedule.
Native Trainer still restores all training state. Exact continuation requires a
**complete standard checkpoint**, unchanged model/loss/environment, and ordinary
uninterrupted epoch progression; custom early-epoch termination and arbitrary
stochastic transforms are not certified. Model-only saves are allowed for export
but do not promise optimizer/RNG continuation. The scheduler does not validate
checkpoint file integrity or impose another optimizer-state format.

`train_samples_per_second` is omitted in this mode because Trainer derives it from
a single nominal batch size. Runtime and optimizer-step throughput remain available.
Trainer's startup batch-size messages are nominal, not actual heterogeneous counts;
use the retained plan above. The inherited `train_loss` keeps native Trainer
semantics and is not a global-objective oracle. CPU two-rank evidence does not
certify GPU/NCCL or mixed-precision numerical behavior.

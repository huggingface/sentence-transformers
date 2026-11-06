"""Pre-training rejection and checkpoint-manifest tests using actual ST objects."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers, SynchronizedMultiDatasetBatchSampler
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer import modules as models

pytest.importorskip("datasets", exc_type=ModuleNotFoundError)
pytest.importorskip("accelerate", exc_type=ModuleNotFoundError)

from datasets import Dataset, DatasetDict


def supports_nullable_batch_metadata():
    """Probe the loader property without guessing a future release number."""
    from accelerate.data_loader import DataLoaderShard
    from torch.utils.data import ConcatDataset

    dataset = ConcatDataset([range(4)])
    plan = SynchronizedMultiDatasetBatchSampler(dataset, batch_sizes=[2], world_size=1)
    loader = DataLoaderShard(dataset, batch_sampler=plan)
    try:
        size = loader.total_batch_size
    except TypeError:
        return False
    assert size is None
    return True


@pytest.fixture
def nullable_batch_metadata():
    # Only positive synchronized loader tests request this fixture. Rejection
    # tests and the existing default training path run with either dependency.
    if not supports_nullable_batch_metadata():
        pytest.skip("Requires Accelerate nullable batch metadata support (huggingface/accelerate#4491)")


def custom_sampler(*args, **kwargs):
    raise AssertionError("Unsupported custom sampler must not be invoked")


def base_args(tmp_path, **overrides):
    options = dict(
        output_dir=str(tmp_path / "output"),
        use_cpu=True,
        report_to=[],
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        synchronized_dataset_batch_sizes={"A": 8, "B": 2},
        multi_dataset_batch_sampler="round_robin",
        save_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
        max_steps=1,
    )
    options.update(overrides)
    return SentenceTransformerTrainingArguments(**options)


@pytest.fixture
def trainer_factory(tmp_path, monkeypatch):
    for name in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
        monkeypatch.setenv(name, "1")

    def create(trainer_class=SentenceTransformerTrainer, dataset=None, **arg_overrides):
        vocabulary = [f"w{i}" for i in range(80)]
        model = SentenceTransformer(
            modules=[
                models.BoW(vocabulary),
                models.Dense(len(vocabulary), 4, bias=False, activation_function=torch.nn.Identity()),
            ],
            device="cpu",
        )
        if dataset is None:
            dataset = DatasetDict(
                {
                    name: Dataset.from_dict(
                        {
                            "anchor": [f"w{i}" for i in range(length)],
                            "positive": [f"w{i + 32}" for i in range(length)],
                            "label": [float(i % 2) for i in range(length)],
                        }
                    )
                    for name, length in [("A", 32), ("B", 8)]
                }
            )
        criterion = (
            {name: losses.CoSENTLoss(model) for name in dataset}
            if isinstance(dataset, dict)
            else losses.CoSENTLoss(model)
        )
        return trainer_class(
            model=model, args=base_args(tmp_path, **arg_overrides), train_dataset=dataset, loss=criterion
        )

    return create


@pytest.mark.parametrize(
    "sampler",
    [BatchSamplers.NO_DUPLICATES, BatchSamplers.NO_DUPLICATES_HASHED, BatchSamplers.GROUP_BY_LABEL, custom_sampler],
)
def test_dynamic_and_custom_samplers_are_rejected_before_training(tmp_path, sampler):
    with pytest.raises(ValueError, match="only BATCH_SAMPLER"):
        base_args(tmp_path, batch_sampler=sampler)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dataloader_drop_last": False},
        {"dataloader_num_workers": 1},
        {"auto_find_batch_size": True},
        {"ignore_data_skip": True},
        {"multi_dataset_batch_sampler": custom_sampler},
        {"synchronized_dataset_batch_sizes": {}},
        {"synchronized_dataset_batch_sizes": {"A": 0, "B": 2}},
        {"synchronized_dataset_batch_sizes": {"A": True, "B": 2}},
        {"synchronized_dataset_batch_sizes": {"A": 8.0, "B": 2}},
        {"synchronized_dataset_batch_sizes": {0: 8, "B": 2}},
        {"synchronized_dataset_batch_sizes": None, "preserve_dataset_batch_frequencies": True},
    ],
)
def test_argument_contract_rejects_unsupported_values(tmp_path, overrides):
    with pytest.raises(ValueError):
        base_args(tmp_path, **overrides)


@pytest.mark.parametrize("config", [{"split_batches": True}, {"dispatch_batches": True}])
def test_loader_modes_rejected_by_real_trainer_preflight(trainer_factory, config):
    trainer = trainer_factory(accelerator_config=config)
    with pytest.raises(ValueError, match="split_batches=False"):
        trainer.get_train_dataloader()


def test_stateful_accelerator_loader_is_rejected_by_trainer_preflight(trainer_factory):
    # Transformers 5.0's AcceleratorConfig does not expose this Accelerate field.
    # Exercise the real accelerator config directly, before any loader is prepared.
    trainer = trainer_factory()
    trainer.accelerator.dataloader_config.use_stateful_dataloader = True
    with pytest.raises(ValueError, match="split_batches=False"):
        trainer.get_train_dataloader()


def test_exact_mapping_and_whole_accumulation_windows_are_checked_before_iteration(trainer_factory):
    trainer = trainer_factory(synchronized_dataset_batch_sizes={"A": 8, "C": 2})
    with pytest.raises(ValueError, match="exactly the training dataset keys"):
        trainer.get_train_dataloader()
    trainer = trainer_factory(gradient_accumulation_steps=3)
    with pytest.raises(ValueError, match="complete gradient accumulation windows"):
        trainer.get_train_dataloader()


def test_plain_dataset_is_not_silently_coerced_to_a_single_task_plan(trainer_factory):
    dataset = Dataset.from_dict({"anchor": ["w1"] * 16, "positive": ["w2"] * 16, "label": [0.0] * 16})
    trainer = trainer_factory(dataset=dataset)
    with pytest.raises(ValueError, match="map-style DatasetDict"):
        trainer.get_train_dataloader()


def test_overridden_child_sampler_is_explicitly_rejected(trainer_factory):
    class OverriddenTrainer(SentenceTransformerTrainer):
        def get_batch_sampler(self, *args, **kwargs):
            raise AssertionError("Must reject rather than silently replace a custom override")

    trainer = trainer_factory(trainer_class=OverriddenTrainer)
    with pytest.raises(ValueError, match="Custom sampler overrides"):
        trainer.get_train_dataloader()


@pytest.mark.parametrize(
    "key",
    [
        "world_size",
        "batch_sizes",
        "task_names",
        "dataset_fingerprints",
        "dataset_sizes",
        "strategy",
        "data_seed",
        "gradient_accumulation_steps",
        "retained_batch_counts",
        "num_microsteps",
    ],
)
def test_changed_checkpoint_manifest_is_rejected_before_model_loading(trainer_factory, tmp_path, key):
    trainer = trainer_factory()
    _, _, expected = trainer._synchronized_batch_components(trainer.train_dataset)
    changed = copy.deepcopy(expected)
    value = changed[key]
    if key == "task_names":
        changed[key] = list(reversed(value))
    elif isinstance(value, list):
        changed[key][0] = value[0] + 1 if isinstance(value[0], int) else value[0] + "_changed"
    elif isinstance(value, int):
        changed[key] = value + 1
    else:
        changed[key] = str(value) + "_changed"
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "synchronized_batch_plan.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed synchronized batch plan"):
        trainer._load_from_checkpoint(str(checkpoint))


def test_missing_resume_manifest_is_rejected(trainer_factory, tmp_path):
    trainer = trainer_factory()
    with pytest.raises(ValueError, match="saved batch-plan manifest"):
        trainer._load_from_checkpoint(str(tmp_path))


def test_synchronized_checkpoint_cannot_be_loaded_with_mode_disabled(trainer_factory, tmp_path):
    trainer = trainer_factory(synchronized_dataset_batch_sizes=None)
    (tmp_path / "synchronized_batch_plan.json").write_text("{}")
    with pytest.raises(ValueError, match="original synchronized dataset batch configuration"):
        trainer._load_from_checkpoint(str(tmp_path))


def test_manifest_reports_actual_counts_and_effective_data_seed(trainer_factory):
    trainer = trainer_factory(seed=17, data_seed=29)
    _, plan, manifest = trainer._synchronized_batch_components(trainer.train_dataset)
    assert manifest["world_size"] == 1
    assert manifest["available_batch_counts"] == [4, 4]
    assert manifest["retained_batch_counts"] == [4, 4]
    assert manifest["num_microsteps"] == plan.num_microsteps == 8
    assert manifest["data_seed"] == plan.seed == 29
    assert manifest["task_names"] == ["A", "B"]


@pytest.mark.parametrize("mutation", ["batch_size", "seed", "mode_disabled"])
def test_save_refuses_configuration_changed_after_loader_preparation(
    trainer_factory, tmp_path, mutation, nullable_batch_metadata
):
    trainer = trainer_factory()
    loader = trainer.get_train_dataloader()
    assert len(loader) == 8
    original_manifest = copy.deepcopy(trainer._synchronized_batch_manifest)
    if mutation == "batch_size":
        trainer.args.synchronized_dataset_batch_sizes = {"A": 4, "B": 2}
    elif mutation == "seed":
        trainer.args.data_seed = original_manifest["data_seed"] + 1
    else:
        trainer.args.synchronized_dataset_batch_sizes = None
    checkpoint = tmp_path / "must-not-be-written"
    with pytest.raises(ValueError):
        trainer._save(str(checkpoint))
    assert not checkpoint.exists(), "Refuse before publishing an inconsistent model checkpoint"
    assert trainer._synchronized_batch_manifest == original_manifest


def test_model_only_saves_are_allowed_without_claiming_continuation(
    trainer_factory, tmp_path, nullable_batch_metadata
):
    trainer = trainer_factory(save_only_model=True, save_strategy="steps", save_steps=1)
    trainer.train()
    checkpoint = tmp_path / "output" / "checkpoint-1"
    assert (checkpoint / "synchronized_batch_plan.json").is_file()
    assert not (checkpoint / "optimizer.pt").exists()


@pytest.mark.parametrize("synchronized", [False, True])
def test_real_train_reporting_is_truthful_and_default_is_unchanged(trainer_factory, synchronized, request):
    if synchronized:
        request.getfixturevalue("nullable_batch_metadata")
    trainer = trainer_factory(synchronized_dataset_batch_sizes={"A": 8, "B": 2} if synchronized else None)
    output = trainer.train()
    assert ("train_samples_per_second" in output.metrics) is not synchronized
    assert ("train_samples_per_second" in trainer.state.log_history[-1]) is not synchronized
    for name in ("train_loss", "train_runtime", "train_steps_per_second"):
        assert name in output.metrics
        assert name in trainer.state.log_history[-1]


def test_matching_manifest_delegates_model_loading_without_inventory_checks(trainer_factory, tmp_path, monkeypatch):
    trainer = trainer_factory()
    _, _, manifest = trainer._synchronized_batch_components(trainer.train_dataset)
    (tmp_path / "synchronized_batch_plan.json").write_text(json.dumps(manifest))

    # No optimizer, RNG, scheduler or Trainer state files. These are owned by
    # native Trainer restoration, not by the schedule-compatibility validator.
    class ReachedNativeModelLoading(Exception):
        pass

    def reached(*args, **kwargs):
        raise ReachedNativeModelLoading

    monkeypatch.setattr(type(trainer.model), "_load_with_module_classes", reached)
    with pytest.raises(ReachedNativeModelLoading):
        trainer._load_from_checkpoint(str(tmp_path))


def test_complete_native_checkpoint_continuation(trainer_factory, tmp_path, monkeypatch, nullable_batch_metadata):
    monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "cpu")
    torch.manual_seed(123)
    trainer = trainer_factory(
        max_steps=4,
        save_strategy="steps",
        save_steps=2,
        lr_scheduler_type="constant",
        output_dir=str(tmp_path / "continuous"),
    )
    trainer.train()
    expected = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    resumed = trainer_factory(
        max_steps=4, save_strategy="no", lr_scheduler_type="constant", output_dir=str(tmp_path / "resumed")
    )
    output = resumed.train(resume_from_checkpoint=str(tmp_path / "continuous" / "checkpoint-2"))
    assert output.global_step == 4
    assert "train_samples_per_second" not in output.metrics
    for key, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)


def test_incompatible_accelerate_metadata_fails_before_iteration(trainer_factory, monkeypatch):
    from accelerate.data_loader import DataLoaderShard

    def old_metadata(self):
        return self.batch_sampler.batch_size * getattr(self.batch_sampler, "num_processes", 1)

    # The pre-fix implementation multiplies None. Do not silently bypass it or
    # advertise a constant size to make incompatible dependency versions work.
    monkeypatch.setattr(DataLoaderShard, "total_batch_size", property(old_metadata))
    trainer = trainer_factory()
    with pytest.raises(ValueError, match="Accelerate variable-batch metadata compatibility fix"):
        trainer.get_train_dataloader()


@pytest.mark.parametrize("strategy", ["round_robin", "proportional"])
@pytest.mark.parametrize("skip", [0, 1, 7])
def test_prepared_loader_epoch_and_native_skip(trainer_factory, nullable_batch_metadata, strategy, skip):
    from accelerate.data_loader import skip_first_batches

    trainer = trainer_factory(multi_dataset_batch_sampler=strategy)
    loader = trainer.get_train_dataloader()

    def snapshot(batches):
        return [
            {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            for batch in batches
        ]

    loader.set_epoch(0)
    first_epoch = snapshot(loader)
    loader.set_epoch(1)
    second_epoch = snapshot(loader)
    assert len(first_epoch) == len(second_epoch) == len(loader) == 8
    assert first_epoch != second_epoch
    resumed = skip_first_batches(loader, skip)
    resumed.set_epoch(1)
    assert len(resumed) == 8 - skip
    assert snapshot(resumed) == second_epoch[skip:]


def test_installed_accelerate_metadata_is_checked_before_training(trainer_factory):
    trainer = trainer_factory()
    if supports_nullable_batch_metadata():
        assert trainer.get_train_dataloader().total_batch_size is None
    else:
        with pytest.raises(ValueError, match="Accelerate variable-batch metadata compatibility fix"):
            trainer.get_train_dataloader()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("strategy", ["round_robin", "proportional"])
@pytest.mark.parametrize("skip", [0, 1, 3])
def test_rank_sharded_epoch_bridge_and_skip(nullable_batch_metadata, rank, strategy, skip):
    from accelerate.data_loader import prepare_data_loader, skip_first_batches
    from torch.utils.data import DataLoader, TensorDataset

    from sentence_transformers.base.sampler import _SynchronizedConcatDataset

    dataset = _SynchronizedConcatDataset([TensorDataset(torch.arange(32)), TensorDataset(torch.arange(100, 108))])
    plan = SynchronizedMultiDatasetBatchSampler(dataset, batch_sizes=[8, 2], world_size=2, strategy=strategy, seed=17)
    dataset.batch_plan = plan
    loader = prepare_data_loader(
        DataLoader(dataset, batch_sampler=plan),
        num_processes=2,
        process_index=rank,
        split_batches=False,
        even_batches=False,
    )
    loader._drop_last = True  # The same global-drop contract recorded by BaseTrainer.
    loader.set_epoch(0)
    first_epoch = [batch[0].tolist() for batch in loader]
    loader.set_epoch(1)
    second_epoch = [batch[0].tolist() for batch in loader]
    assert len(first_epoch) == len(second_epoch) == len(loader) == 4
    assert first_epoch != second_epoch
    assert sorted(map(len, second_epoch)) == [2, 2, 8, 8]
    resumed = skip_first_batches(loader, skip)
    resumed.set_epoch(1)
    assert plan.epoch == 1
    assert len(resumed) == 4 - skip
    assert [batch[0].tolist() for batch in resumed] == second_epoch[skip:]

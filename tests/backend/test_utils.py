from __future__ import annotations

import shutil
from pathlib import Path

import huggingface_hub
import numpy as np
import pytest

try:
    import onnx
    import onnxruntime
    from onnx import TensorProto, helper, numpy_helper
    from onnx.external_data_helper import _get_all_tensors, uses_external_data
except ImportError:
    pytest.skip("onnx is not available", allow_module_level=True)

from sentence_transformers.backend.utils import save_or_push_to_hub_model

FILE_NAME = "model_O1.onnx"


def make_weights(seed: int, name: str) -> TensorProto:
    return numpy_helper.from_array(np.arange(16, dtype=np.float32).reshape(4, 4) + seed, name=name)


def make_model(initializers: int = 1, constants: int = 0, in_subgraph: bool = False) -> onnx.ModelProto:
    """Build a tiny model that adds a few weight tensors to its input.

    The weights are held as initializers of the graph, as values of Constant node attributes, or as
    initializers of the two subgraphs of an If node, which covers where a model can keep a tensor.
    """
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 4])]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 4])]
    opset = [helper.make_opsetid("", 17)]

    if in_subgraph:
        branches = {
            f"{branch}_branch": helper.make_graph(
                [helper.make_node("Add", ["x", branch], ["out"])],
                f"{branch}_graph",
                [],
                [helper.make_tensor_value_info("out", TensorProto.FLOAT, [4, 4])],
                initializer=[make_weights(seed, branch)],
            )
            for seed, branch in enumerate(["then", "else"])
        }
        inputs.append(helper.make_tensor_value_info("cond", TensorProto.BOOL, []))
        graph = helper.make_graph([helper.make_node("If", ["cond"], ["y"], **branches)], "g", inputs, outputs)
        return helper.make_model(graph, opset_imports=opset, ir_version=10)

    names = [f"W{index}" for index in range(initializers)] + [f"C{index}" for index in range(constants)]
    nodes, initializer_tensors, previous = [], [], "x"
    for index, name in enumerate(names):
        output = "y" if index == len(names) - 1 else f"y{index}"
        if name.startswith("C"):
            nodes.append(helper.make_node("Constant", [], [f"{name}_out"], value=make_weights(index, name)))
            operand = f"{name}_out"
        else:
            initializer_tensors.append(make_weights(index, name))
            operand = name
        nodes.append(helper.make_node("Add", [previous, operand], [output]))
        previous = output

    graph = helper.make_graph(nodes, "g", inputs, outputs, initializer=initializer_tensors)
    return helper.make_model(graph, opset_imports=opset, ir_version=10)


def save_model(path: Path, model: onnx.ModelProto | None = None, **external_data_kwargs) -> None:
    """Save a model, with its weights either inside the ONNX file or in files beside it.

    A model over the 2GB protobuf limit gets that split automatically. `size_threshold=0` asks for it
    regardless of size, so the same layout can be tested without a multi-gigabyte model.
    """
    if external_data_kwargs:
        external_data_kwargs = {"size_threshold": 0, "convert_attribute": True, **external_data_kwargs}
    onnx.save_model(
        model if model is not None else make_model(),
        path.as_posix(),
        save_as_external_data=bool(external_data_kwargs),
        **external_data_kwargs,
    )


def get_locations(path: Path) -> set[str]:
    """The external data locations that a model refers to."""
    model = onnx.load(path.as_posix(), load_external_data=False)
    return {
        entry.value
        for tensor in _get_all_tensors(model)
        if uses_external_data(tensor)
        for entry in tensor.external_data
        if entry.key == "location"
    }


def set_locations(path: Path, locations: dict[str, str]) -> None:
    """Point a model at other external data locations, leaving the offsets into them alone."""
    model = onnx.load(path.as_posix(), load_external_data=False)
    for tensor in _get_all_tensors(model):
        for entry in tensor.external_data if uses_external_data(tensor) else []:
            if entry.key == "location" and entry.value in locations:
                entry.value = locations[entry.value]
    onnx.save(model, path.as_posix())


def export_to(destination: Path, export_function) -> set[str]:
    """Run save_or_push_to_hub_model against a local directory and return the files it wrote."""
    save_or_push_to_hub_model(
        export_function=export_function,
        export_function_name="export_optimized_onnx_model",
        config="O1",
        model_name_or_path=destination.as_posix(),
        push_to_hub=False,
        file_suffix="O1",
        backend="onnx",
        model=None,
    )
    onnx_dir = destination / "onnx"
    return {path.relative_to(onnx_dir).as_posix() for path in onnx_dir.rglob("*") if path.is_file()}


def push_to_hub(destination: Path, export_function, monkeypatch) -> set[str]:
    """Run save_or_push_to_hub_model with push_to_hub=True and return the files it hands to the Hub.

    The folder is uploaded whole and then discarded, so it is copied aside to be looked at.
    """

    def upload_folder(folder_path, **kwargs):
        shutil.copytree(folder_path, destination, dirs_exist_ok=True)

    monkeypatch.setattr(huggingface_hub, "upload_folder", upload_folder)
    save_or_push_to_hub_model(
        export_function=export_function,
        export_function_name="export_optimized_onnx_model",
        config="O1",
        model_name_or_path="sentence-transformers-testing/some-model",
        push_to_hub=True,
        file_suffix="O1",
        backend="onnx",
        model=None,
    )
    return {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}


def assert_runs(model_path: Path, **inputs) -> None:
    """A model is only usable if the files holding its weights came along, so load it and run it."""
    onnx.load(model_path.as_posix())
    session = onnxruntime.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"x": np.zeros((4, 4), dtype=np.float32), **inputs})
    assert np.isfinite(outputs[0]).all()


def test_onnx_export_keeps_external_data(tmp_path: Path) -> None:
    """The weights of a model over 2GB live beside the ONNX file, so they have to be saved with it."""

    def export_function(save_dir):
        save_model(Path(save_dir, FILE_NAME), location=f"{FILE_NAME}.data")

    assert export_to(tmp_path, export_function) == {FILE_NAME, f"{FILE_NAME}_data"}
    assert_runs(tmp_path / "onnx" / FILE_NAME)


def test_onnx_export_without_external_data(tmp_path: Path) -> None:
    """A model that fits within the protobuf limit has no separate weights file to look for."""

    def export_function(save_dir):
        save_model(Path(save_dir, FILE_NAME))

    assert export_to(tmp_path, export_function) == {FILE_NAME}
    assert_runs(tmp_path / "onnx" / FILE_NAME)


def test_onnx_export_keeps_external_data_of_node_attributes(tmp_path: Path) -> None:
    """The tensors held by node attributes get externalized as well, and those are not initializers."""

    def export_function(save_dir):
        save_model(Path(save_dir, FILE_NAME), make_model(constants=1), all_tensors_to_one_file=False)

    assert export_to(tmp_path, export_function) == {FILE_NAME, "W0", "C0"}
    assert_runs(tmp_path / "onnx" / FILE_NAME)


def test_onnx_export_keeps_external_data_of_subgraphs(tmp_path: Path) -> None:
    """The branches of an If, Loop or Scan node are graphs of their own, with their own weights."""

    def export_function(save_dir):
        save_model(Path(save_dir, FILE_NAME), make_model(in_subgraph=True), all_tensors_to_one_file=False)

    assert export_to(tmp_path, export_function) == {FILE_NAME, "then", "else"}
    assert_runs(tmp_path / "onnx" / FILE_NAME, cond=np.array(True))


def test_onnx_export_keeps_external_data_spread_over_several_files(tmp_path: Path) -> None:
    """Weights can also be written one file per tensor, under names taken from the tensor names."""

    def export_function(save_dir):
        save_model(Path(save_dir, FILE_NAME), make_model(initializers=2), all_tensors_to_one_file=False)

    assert export_to(tmp_path, export_function) == {FILE_NAME, "W0", "W1"}
    assert_runs(tmp_path / "onnx" / FILE_NAME)


def test_onnx_export_keeps_external_data_in_subdirectories(tmp_path: Path) -> None:
    """Weights may sit in a subdirectory, and the model keeps pointing at them only if that is kept.

    Optimum writes a plain file name beside the model, so this layout is built by hand. It is the one
    the rename leaves alone, and two files sharing a name is what makes keeping the directories more
    than cosmetic, since flattening would lose one.
    """

    def export_function(save_dir):
        path = Path(save_dir, FILE_NAME)
        save_model(path, make_model(initializers=2), all_tensors_to_one_file=False)
        for tensor_name, directory in [("W0", "first"), ("W1", "second")]:
            Path(save_dir, directory).mkdir()
            shutil.move(Path(save_dir, tensor_name), Path(save_dir, directory, "weights.bin"))
        set_locations(path, {"W0": "first/weights.bin", "W1": "second/weights.bin"})

    assert export_to(tmp_path, export_function) == {FILE_NAME, "first/weights.bin", "second/weights.bin"}
    assert_runs(tmp_path / "onnx" / FILE_NAME)


def test_onnx_export_renames_external_data_for_the_hub(tmp_path: Path) -> None:
    """ONNX Runtime writes `<model>.onnx.data`, but only `<model>.onnx_data` is fetched from the Hub."""

    def export_function(save_dir):
        save_model(Path(save_dir, FILE_NAME), location=f"{FILE_NAME}.data")

    assert export_to(tmp_path, export_function) == {FILE_NAME, f"{FILE_NAME}_data"}
    assert get_locations(tmp_path / "onnx" / FILE_NAME) == {f"{FILE_NAME}_data"}
    assert_runs(tmp_path / "onnx" / FILE_NAME)


def test_onnx_export_keeps_the_hub_external_data_name(tmp_path: Path) -> None:
    """Optimum already writes the name the Hub needs when exporting, so that model is left alone."""

    def export_function(save_dir):
        save_model(Path(save_dir, FILE_NAME), location=f"{FILE_NAME}_data")

    assert export_to(tmp_path, export_function) == {FILE_NAME, f"{FILE_NAME}_data"}
    assert get_locations(tmp_path / "onnx" / FILE_NAME) == {f"{FILE_NAME}_data"}
    assert_runs(tmp_path / "onnx" / FILE_NAME)


def test_onnx_push_to_hub_uploads_external_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model whose weights are left behind is uploaded broken, which is what the Hub then serves."""

    def export_function(save_dir):
        save_model(Path(save_dir, FILE_NAME), location=f"{FILE_NAME}.data")

    assert push_to_hub(tmp_path, export_function, monkeypatch) == {FILE_NAME, f"{FILE_NAME}_data"}
    assert get_locations(tmp_path / FILE_NAME) == {f"{FILE_NAME}_data"}
    assert_runs(tmp_path / FILE_NAME)


def test_onnx_push_to_hub_uploads_external_data_of_subgraphs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Weights spread over several files keep their names, since no single one is the one to fetch."""

    def export_function(save_dir):
        save_model(Path(save_dir, FILE_NAME), make_model(in_subgraph=True), all_tensors_to_one_file=False)

    assert push_to_hub(tmp_path, export_function, monkeypatch) == {FILE_NAME, "then", "else"}
    assert_runs(tmp_path / FILE_NAME, cond=np.array(True))

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi
from onnx import TensorProto, helper, numpy_helper
from pydantic import BaseModel, Field

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("onnx requires numpy to build tensor constants") from exc

load_dotenv()

ROOT = Path(__file__).resolve().parent
CLIENT_DIST = ROOT / "client" / "dist"
BANNED_OPS = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function"}
SUPPORTED_OPS = {
    "Input",
    "Output",
    "Constant",
    "Cast",
    "Identity",
    "Equal",
    "Greater",
    "Less",
    "Not",
    "And",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "ReduceSum",
    "ArgMax",
    "Where",
}
CANVAS_SHAPE = [1, 1, 30, 30]
TASK_ID_RE = re.compile(r"^task\d{3}$")

app = FastAPI(title="NeuroGolf Lab")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExportPayload(BaseModel):
    projectName: str = "neurogolf-graph"
    taskId: str = "task000"
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]] = Field(default_factory=list)
    trainingPairs: list[dict[str, Any]] = Field(default_factory=list)


class ValidationError(Exception):
    pass


def _node_id(node: dict[str, Any]) -> str:
    value = str(node.get("id") or node.get("data", {}).get("id") or "").strip()
    if not value:
        raise ValueError("Every graph node must have a stable id")
    return value


def _op_type(node: dict[str, Any]) -> str:
    data = node.get("data") or {}
    return str(data.get("opType") or data.get("label") or node.get("type") or "").strip()


def _task_id(payload: ExportPayload) -> str:
    task_id = payload.taskId.strip().lower()
    if not TASK_ID_RE.match(task_id):
        raise ValueError("taskId must match taskXXX, for example task010")
    return task_id


def _parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "," in text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _raw_attrs(data: dict[str, Any]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if isinstance(data.get("attrs"), dict):
        attrs.update(data["attrs"])
    attrs_text = data.get("attrsText")
    if isinstance(attrs_text, str) and attrs_text.strip():
        try:
            parsed = json.loads(attrs_text)
            if not isinstance(parsed, dict):
                raise ValueError("attrsText must be a JSON object")
            attrs.update(parsed)
        except json.JSONDecodeError as exc:
            raise ValueError(f"attrsText is not valid JSON: {exc}") from exc
    return {str(key).strip(): _parse_literal(value) for key, value in attrs.items() if value not in ("", None)}


def _shape(data: dict[str, Any] | None, default: list[int] | None = None) -> list[int]:
    data = data or {}
    raw = data.get("shape") if data.get("shape") not in ("", None) else (default or CANVAS_SHAPE)
    raw = _parse_literal(raw)
    if isinstance(raw, str):
        raw = [int(part.strip()) for part in raw.replace("x", ",").split(",") if part.strip()]
    if not isinstance(raw, list) or not raw or not all(isinstance(item, int) and item > 0 for item in raw):
        raise ValueError("All tensors must have statically defined positive integer shapes")
    return raw


def _onnx_attrs(op: str, attrs: dict[str, Any]) -> dict[str, Any]:
    if op == "Cast":
        return {"to": int(attrs.get("to", TensorProto.FLOAT))}
    if op == "ArgMax":
        return {"axis": int(attrs.get("axis", 1)), "keepdims": int(attrs.get("keepdims", 1))}
    if op == "ReduceSum":
        return {"keepdims": int(attrs.get("keepdims", 1))}
    return {}


def _output_shape_for_reduction(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    axes = attrs.get("axes")
    keepdims = int(attrs.get("keepdims", 1))
    if axes is None:
        axes = list(range(len(input_shape)))
    if isinstance(axes, int):
        axes = [axes]
    axes = [(axis + len(input_shape)) % len(input_shape) for axis in axes]
    if keepdims:
        return [1 if idx in axes else dim for idx, dim in enumerate(input_shape)]
    return [dim for idx, dim in enumerate(input_shape) if idx not in axes] or [1]


def _output_shape_for_argmax(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    axis = int(attrs.get("axis", 1))
    keepdims = int(attrs.get("keepdims", 1))
    axis = (axis + len(input_shape)) % len(input_shape)
    if keepdims:
        return [1 if idx == axis else dim for idx, dim in enumerate(input_shape)]
    return [dim for idx, dim in enumerate(input_shape) if idx != axis] or [1]


def _tensor_type_for_cast(attrs: dict[str, Any]) -> int:
    return int(attrs.get("to", TensorProto.FLOAT))


def _validate_graph(payload: ExportPayload) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if not payload.nodes:
        raise ValueError("Graph must contain at least one node")

    by_id: dict[str, dict[str, Any]] = {}
    for node in payload.nodes:
        node_id = _node_id(node)
        if node_id in by_id:
            raise ValueError(f"Duplicate node id: {node_id}")
        op = _op_type(node)
        if op in BANNED_OPS:
            raise ValueError(f"Banned ONNX operation(s): {op}")
        if op not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported ONNX operation: {op}")
        by_id[node_id] = node

    incoming: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in by_id}
    for edge in payload.edges:
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if source not in by_id:
            raise ValueError(f"Malformed graph: edge source {source!r} does not exist")
        if target not in by_id:
            raise ValueError(f"Malformed graph: edge target {target!r} does not exist")
        if source == target:
            raise ValueError(f"Malformed graph: self-edge on {source!r}")
        incoming[target].append(edge)

    if not any(_op_type(node) == "Input" for node in by_id.values()):
        raise ValueError("Graph must contain at least one Input node")
    if not any(_op_type(node) == "Output" for node in by_id.values()):
        raise ValueError("Graph must contain at least one Output node")

    sorted_nodes = _topological_sort(by_id, incoming)
    return by_id, incoming, sorted_nodes


def _topological_sort(by_id: dict[str, dict[str, Any]], incoming: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    deps = {node_id: {str(edge["source"]) for edge in edges} for node_id, edges in incoming.items()}
    ready = sorted(node_id for node_id, node_deps in deps.items() if not node_deps)
    ordered_ids: list[str] = []

    while ready:
        node_id = ready.pop(0)
        ordered_ids.append(node_id)
        for other_id in sorted(deps):
            if node_id in deps[other_id]:
                deps[other_id].remove(node_id)
                if not deps[other_id] and other_id not in ordered_ids and other_id not in ready:
                    ready.append(other_id)
        ready.sort()

    if len(ordered_ids) != len(by_id):
        raise ValueError("Malformed graph: cycle detected or unreachable dependency")
    return [by_id[node_id] for node_id in ordered_ids]


def _incoming_ids(node_id: str, incoming: dict[str, list[dict[str, Any]]]) -> list[str]:
    edges = sorted(
        incoming.get(node_id, []),
        key=lambda edge: (str(edge.get("targetHandle") or ""), str(edge.get("source") or ""), str(edge.get("id") or "")),
    )
    return [str(edge["source"]) for edge in edges]


def _expect_inputs(op: str, node_id: str, ids: list[str]) -> None:
    required = {
        "Cast": 1,
        "Identity": 1,
        "Not": 1,
        "ReduceSum": 1,
        "ArgMax": 1,
        "Equal": 2,
        "Greater": 2,
        "Less": 2,
        "And": 2,
        "Add": 2,
        "Sub": 2,
        "Mul": 2,
        "Div": 2,
        "Where": 3,
        "Output": 1,
    }
    expected = required.get(op)
    if expected is not None and len(ids) != expected:
        raise ValueError(f"{op} node {node_id!r} requires {expected} input edge(s), got {len(ids)}")


def _constant_array(data: dict[str, Any], shape: list[int]) -> np.ndarray:
    values = data.get("values", data.get("value"))
    if values is None:
        return np.zeros(shape, dtype=np.float32)
    array = np.asarray(_parse_literal(values), dtype=np.float32)
    if array.size == 1 and int(np.prod(shape)) != 1:
        return np.full(shape, float(array.reshape(-1)[0]), dtype=np.float32)
    try:
        return array.reshape(shape).astype(np.float32)
    except ValueError as exc:
        raise ValueError(f"Constant values cannot be reshaped to {shape}") from exc


def compile_graph(payload: ExportPayload) -> onnx.ModelProto:
    _task_id(payload)
    _by_id, incoming, sorted_nodes = _validate_graph(payload)

    initializers = []
    value_infos = []
    onnx_nodes = []
    graph_outputs = []
    graph_inputs = []
    tensor_name: dict[str, str] = {}
    tensor_shape: dict[str, list[int]] = {}
    tensor_type: dict[str, int] = {}

    for node in sorted_nodes:
        node_id = _node_id(node)
        data = node.get("data") or {}
        op = _op_type(node)
        attrs = _raw_attrs(data)
        output_name = f"{node_id}_out"

        if op == "Input":
            shape = _shape(data, CANVAS_SHAPE)
            if shape != CANVAS_SHAPE:
                raise ValueError(f"Input node {node_id!r} must use static shape {CANVAS_SHAPE}")
            tensor_name[node_id] = output_name
            tensor_shape[node_id] = shape
            tensor_type[node_id] = TensorProto.FLOAT
            graph_inputs.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape))
            continue

        input_ids = _incoming_ids(node_id, incoming)

        if op == "Output":
            _expect_inputs(op, node_id, input_ids)
            source_id = input_ids[0]
            source_name = tensor_name[source_id]
            graph_outputs.append(helper.make_tensor_value_info(source_name, tensor_type[source_id], tensor_shape[source_id]))
            continue

        if op == "Constant":
            shape = _shape(data, CANVAS_SHAPE)
            array = _constant_array(data, shape)
            tensor = numpy_helper.from_array(array, name=f"{node_id}_value")
            tensor_name[node_id] = output_name
            tensor_shape[node_id] = shape
            tensor_type[node_id] = TensorProto.FLOAT
            onnx_nodes.append(helper.make_node("Constant", inputs=[], outputs=[output_name], name=node_id, value=tensor))
            value_infos.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape))
            continue

        _expect_inputs(op, node_id, input_ids)
        inputs = [tensor_name[source_id] for source_id in input_ids]
        input_shapes = [tensor_shape[source_id] for source_id in input_ids]
        input_types = [tensor_type[source_id] for source_id in input_ids]
        shape = _shape(data, input_shapes[0])
        out_type = input_types[0]
        node_inputs = inputs

        if op in {"Equal", "Greater", "Less", "And", "Add", "Sub", "Mul", "Div", "Where"}:
            for idx, input_shape in enumerate(input_shapes, start=1):
                if input_shape != input_shapes[0]:
                    raise ValueError(f"{op} node {node_id!r} input {idx} shape {input_shape} does not match {input_shapes[0]}")
            shape = input_shapes[0]
        if op in {"Equal", "Greater", "Less"}:
            out_type = TensorProto.BOOL
        elif op in {"Not", "And"}:
            out_type = TensorProto.BOOL
        elif op == "Cast":
            out_type = _tensor_type_for_cast(attrs)
        elif op == "Where":
            if input_types[1] != input_types[2]:
                raise ValueError(f"Where node {node_id!r} true/false inputs must have matching tensor types")
            out_type = input_types[1]
        elif op == "ReduceSum":
            axes = attrs.get("axes")
            shape = _output_shape_for_reduction(input_shapes[0], attrs)
            if axes is not None:
                if isinstance(axes, int):
                    axes = [axes]
                axes_name = f"{node_id}_axes"
                initializers.append(helper.make_tensor(axes_name, TensorProto.INT64, [len(axes)], [int(axis) for axis in axes]))
                node_inputs = [inputs[0], axes_name]
        elif op == "ArgMax":
            shape = _output_shape_for_argmax(input_shapes[0], attrs)
            out_type = TensorProto.INT64

        tensor_name[node_id] = output_name
        tensor_shape[node_id] = shape
        tensor_type[node_id] = out_type
        onnx_nodes.append(helper.make_node(op, inputs=node_inputs, outputs=[output_name], name=node_id, **_onnx_attrs(op, attrs)))
        value_infos.append(helper.make_tensor_value_info(output_name, out_type, shape))

    if not graph_outputs:
        raise ValueError("Graph must contain at least one connected Output node")

    graph = helper.make_graph(
        onnx_nodes,
        "NeuroGolfLabGraph",
        graph_inputs,
        graph_outputs,
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(graph, producer_name="neurogolf-lab", opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def save_model(model: onnx.ModelProto, payload: ExportPayload) -> Path:
    task_id = _task_id(payload)
    out_dir = Path(tempfile.mkdtemp(prefix="neurogolf_"))
    out_path = out_dir / f"{task_id}.onnx"
    onnx.save(model, out_path)
    return out_path


def _arc_grid_to_canvas(value: Any) -> tuple[np.ndarray, tuple[int, int]]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        height, width = arr.shape
        tensor = arr.reshape(1, 1, height, width)
    elif arr.ndim == 4:
        tensor = arr.astype(np.float32)
        height, width = tensor.shape[-2:]
    else:
        raise ValueError(f"expected a 2D ARC grid or NCHW tensor, got rank {arr.ndim}")
    if height > 30 or width > 30:
        raise ValueError(f"ARC grid shape {[height, width]} exceeds 30x30 canvas")
    canvas = np.zeros(CANVAS_SHAPE, dtype=np.float32)
    canvas[:, :, :height, :width] = tensor[:, :, :height, :width]
    return canvas, (height, width)


def _first_bad_index(actual: np.ndarray, expected: np.ndarray) -> list[int]:
    bad = np.argwhere(actual != expected)
    if bad.size == 0:
        return []
    idx = bad[0].tolist()
    return idx[-2:] if len(idx) >= 2 else idx


def _assert_color_bounds(label: str, tensor: np.ndarray) -> None:
    values = np.asarray(tensor)
    if values.dtype.kind == "b":
        raise ValidationError(f"Phase 3 Color Bounds failed: {label} produced boolean values, not ARC color integers")
    if not np.all(np.isfinite(values)):
        idx = np.argwhere(~np.isfinite(values))[0].tolist()
        raise ValidationError(f"Phase 3 Color Bounds failed: {label} produced non-finite value at index {idx[-2:]}")
    if not np.all(values == np.round(values)):
        idx = np.argwhere(values != np.round(values))[0].tolist()
        raise ValidationError(f"Phase 3 Color Bounds failed: {label} produced non-integer value at index {idx[-2:]}")
    if not np.all((values >= 0) & (values <= 9)):
        idx = np.argwhere((values < 0) | (values > 9))[0].tolist()
        bad_value = values[tuple(idx)]
        raise ValidationError(f"Phase 3 Color Bounds failed: {label} produced color {bad_value:g} at index {idx[-2:]}")


def _run_session(session: ort.InferenceSession, source_grid: Any) -> tuple[np.ndarray, tuple[int, int]]:
    inputs = session.get_inputs()
    if not inputs:
        raise ValidationError("Phase 1 Strict Equivalence failed: compiled model has no inputs")
    canvas, region = _arc_grid_to_canvas(source_grid)
    feed = {}
    for input_meta in inputs:
        if list(input_meta.shape) != CANVAS_SHAPE:
            raise ValidationError(f"Model input {input_meta.name} has shape {input_meta.shape}, expected {CANVAS_SHAPE}")
        feed[input_meta.name] = canvas
    return session.run(None, feed)[0], region


def validate_model(model: onnx.ModelProto, payload: ExportPayload) -> dict[str, str]:
    if not payload.trainingPairs:
        raise ValidationError("Phase 1 Strict Equivalence failed: no ARC training pairs were supplied")

    try:
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ValidationError(f"Phase 1 Strict Equivalence failed: ONNX Runtime could not load model: {exc}") from exc

    strict_outputs: list[np.ndarray] = []
    canvas_outputs: list[np.ndarray] = []

    for index, pair in enumerate(payload.trainingPairs, start=1):
        expected_key = "output" if "output" in pair else "target"
        if "input" not in pair or expected_key not in pair:
            raise ValidationError(f"Phase 1 Strict Equivalence failed: Train {index} is missing input or output grid")
        try:
            actual, _input_region = _run_session(session, pair["input"])
            expected_canvas, (height, width) = _arc_grid_to_canvas(pair[expected_key])
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"Phase 1 Strict Equivalence failed: Train {index} runtime error: {exc}") from exc
        if actual.ndim < 2 or actual.shape[-2] < height or actual.shape[-1] < width:
            raise ValidationError(
                f"Phase 1 Strict Equivalence failed: Train {index} output shape {list(actual.shape)} "
                f"cannot cover expected window {[height, width]}"
            )
        actual_window = actual[..., :height, :width]
        expected_window = expected_canvas[..., :height, :width]
        if not np.array_equal(actual_window, expected_window):
            bad_index = _first_bad_index(actual_window, expected_window)
            raise ValidationError(f"Phase 1 Strict Equivalence failed: Train {index} output mismatched at index {bad_index}")
        strict_outputs.append(actual)

    for index, pair in enumerate(payload.trainingPairs, start=1):
        try:
            actual, _region = _run_session(session, pair["input"])
            canvas_outputs.append(actual)
        except Exception as exc:
            raise ValidationError(f"Phase 2 Canvas Test failed: Train {index} 30x30 canvas runtime error: {exc}") from exc

    for index, tensor in enumerate(strict_outputs, start=1):
        _assert_color_bounds(f"Train {index}", tensor)
    for index, tensor in enumerate(canvas_outputs, start=1):
        _assert_color_bounds(f"Canvas Train {index}", tensor)

    return {"train": "passed", "shape": "passed", "colors": "passed"}


@app.post("/api/export")
def export_onnx(payload: ExportPayload):
    try:
        task_id = _task_id(payload)
        model = compile_graph(payload)
        validation = validate_model(model, payload)
        artifact = save_model(model, payload)
        token = os.getenv("HF_TOKEN")
        repo_id = os.getenv("HF_REPO_ID")
        if not token or not repo_id:
            raise ValueError("HF_TOKEN and HF_REPO_ID are required")
        api = HfApi(token=token)
        remote_path = f"{task_id}.onnx"
        api.upload_file(path_or_fileobj=str(artifact), path_in_repo=remote_path, repo_id=repo_id, repo_type="model")
        return {"status": "passed", "artifact": artifact.name, "repo": repo_id, "path": remote_path, "validation": validation}
    except ValidationError as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})


if CLIENT_DIST.exists():
    app.mount("/", StaticFiles(directory=str(CLIENT_DIST), html=True), name="static")

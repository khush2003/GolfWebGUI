import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi
from onnx import TensorProto, helper, numpy_helper
from pydantic import BaseModel, Field

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("onnx requires numpy to build tensor constants") from exc

load_dotenv()


def _ort_providers() -> list[str]:
    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


ROOT = Path(__file__).resolve().parent
CLIENT_DIST = ROOT / "client" / "dist"
CLIENT_PUBLIC = ROOT / "client" / "public"
IMPORT_DIR = Path(os.getenv("IMPORT_DIR", "/tmp/neurogolf_imports"))
IMPORT_DIR.mkdir(parents=True, exist_ok=True)
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
    "GreaterOrEqual",
    "LessOrEqual",
    "Not",
    "And",
    "Or",
    "Xor",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Mod",
    "Min",
    "Max",
    "Sum",
    "Relu",
    "Abs",
    "Neg",
    "Floor",
    "Clip",
    "Sign",
    "Sqrt",
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "ArgMax",
    "Where",
    "Slice",
    "Pad",
    "Concat",
    "Transpose",
    "Tile",
    "Resize",
    "Conv",
    "RowIndex",
    "ColIndex",
}
CANVAS_SHAPE = [1, 1, 30, 30]
TASK_ID_RE = re.compile(r"^task\d{3}$")
INPUT_SLOT_ORDER = {
    "Cast": ["input"],
    "Identity": ["input"],
    "Not": ["input"],
    "ReduceSum": ["input"],
    "ArgMax": ["input"],
    "Slice": ["input"],
    "Pad": ["input"],
    "Transpose": ["input"],
    "Tile": ["input"],
    "Resize": ["input"],
    "Conv": ["input"],
    "Output": ["input"],
    "Equal": ["a", "b"],
    "Greater": ["a", "b"],
    "Less": ["a", "b"],
    "GreaterOrEqual": ["a", "b"],
    "LessOrEqual": ["a", "b"],
    "And": ["a", "b"],
    "Or": ["a", "b"],
    "Xor": ["a", "b"],
    "Add": ["a", "b"],
    "Sub": ["a", "b"],
    "Mul": ["a", "b"],
    "Div": ["a", "b"],
    "Mod": ["a", "b"],
    "Min": ["a", "b"],
    "Max": ["a", "b"],
    "Sum": ["a", "b"],
    "Relu": ["input"],
    "Abs": ["input"],
    "Neg": ["input"],
    "Floor": ["input"],
    "Clip": ["input"],
    "Sign": ["input"],
    "Sqrt": ["input"],
    "ReduceMax": ["input"],
    "ReduceMin": ["input"],
    "Where": ["condition", "true", "false"],
    "Concat": ["a", "b"],
}

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


class RunPayload(ExportPayload):
    inputGrid: Any | None = None


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


def _hf_token_username(api: HfApi) -> str:
    info = api.whoami()
    username = (info.get("name") or "").strip()
    if not username:
        raise ValueError("Could not verify Hugging Face token owner")
    return username


def _assert_hf_repo_matches_token(api: HfApi, repo_id: str) -> None:
    username = _hf_token_username(api)
    namespace = repo_id.split("/", 1)[0].strip()
    if namespace != username:
        raise ValueError(
            f"HF_REPO_ID must be under your Hugging Face account ({username}/...), "
            f"but it is set to {repo_id!r}"
        )


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
    if op in {"ReduceMax", "ReduceMin"}:
        result = {"keepdims": int(attrs.get("keepdims", 1))}
        if "axes" in attrs:
            axes = attrs["axes"] if isinstance(attrs["axes"], list) else [attrs["axes"]]
            result["axes"] = [int(axis) for axis in axes]
        return result
    if op == "Concat":
        return {"axis": int(attrs.get("axis", 1))}
    if op == "Transpose":
        return {"perm": [int(item) for item in attrs.get("perm", [0, 1, 3, 2])]}
    if op == "Pad":
        return {"mode": str(attrs.get("mode", "constant"))}
    if op == "Resize":
        return {
            "mode": str(attrs.get("mode", "nearest")),
            "coordinate_transformation_mode": str(attrs.get("coordinate_transformation_mode", "asymmetric")),
            "nearest_mode": str(attrs.get("nearest_mode", "floor")),
        }
    if op == "Conv":
        result: dict[str, Any] = {}
        if "pads" in attrs:
            result["pads"] = [int(item) for item in attrs["pads"]]
        if "strides" in attrs:
            result["strides"] = [int(item) for item in attrs["strides"]]
        if "dilations" in attrs:
            result["dilations"] = [int(item) for item in attrs["dilations"]]
        return result
    if op == "Mod":
        return {"fmod": int(attrs.get("fmod", 0))}
    return {}


def _int_list(value: Any, default: list[int]) -> list[int]:
    parsed = _parse_literal(value) if value not in ("", None) else default
    if isinstance(parsed, int):
        parsed = [parsed]
    if isinstance(parsed, str):
        parsed = [int(part.strip()) for part in parsed.replace("x", ",").split(",") if part.strip()]
    if not isinstance(parsed, list) or not all(isinstance(item, int) for item in parsed):
        raise ValueError("expected an integer list")
    return parsed


def _axis(axis: int, rank: int) -> int:
    return (axis + rank) % rank


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


def _output_shape_for_slice(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    axes = _int_list(attrs.get("axes"), list(range(len(input_shape))))
    starts = _int_list(attrs.get("starts"), [0 for _ in axes])
    ends = _int_list(attrs.get("ends"), [input_shape[_axis(axis, len(input_shape))] for axis in axes])
    steps = _int_list(attrs.get("steps"), [1 for _ in axes])
    if not (len(starts) == len(ends) == len(axes) == len(steps)):
        raise ValueError("Slice starts, ends, axes, and steps must have the same length")
    shape = list(input_shape)
    for start, end, axis, step in zip(starts, ends, axes, steps):
        axis = _axis(axis, len(input_shape))
        if step <= 0:
            raise ValueError("Slice steps must be positive")
        dim = input_shape[axis]
        start = max(0, min(dim, start if start >= 0 else dim + start))
        end = max(0, min(dim, end if end >= 0 else dim + end))
        shape[axis] = max(0, (end - start + step - 1) // step)
        if shape[axis] <= 0:
            raise ValueError("Slice output dimensions must be positive")
    return shape


def _output_shape_for_pad(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    pads = _int_list(attrs.get("pads"), [0, 0, 0, 0, 0, 0, 0, 0])
    if len(pads) != 2 * len(input_shape):
        raise ValueError(f"Pad pads must have {2 * len(input_shape)} values")
    return [dim + pads[idx] + pads[idx + len(input_shape)] for idx, dim in enumerate(input_shape)]


def _output_shape_for_concat(input_shapes: list[list[int]], attrs: dict[str, Any]) -> list[int]:
    axis = _axis(int(attrs.get("axis", 1)), len(input_shapes[0]))
    shape = list(input_shapes[0])
    shape[axis] = 0
    for idx, input_shape in enumerate(input_shapes, start=1):
        if len(input_shape) != len(shape):
            raise ValueError(f"Concat input {idx} rank does not match")
        for dim_index, dim in enumerate(input_shape):
            if dim_index != axis and dim != input_shapes[0][dim_index]:
                raise ValueError(f"Concat input {idx} shape {input_shape} is incompatible on axis {axis}")
        shape[axis] += input_shape[axis]
    return shape


def _output_shape_for_transpose(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    perm = _int_list(attrs.get("perm"), [0, 1, 3, 2])
    if sorted(perm) != list(range(len(input_shape))):
        raise ValueError("Transpose perm must contain every input axis exactly once")
    return [input_shape[idx] for idx in perm]


def _output_shape_for_tile(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    repeats = _int_list(attrs.get("repeats"), [1 for _ in input_shape])
    if len(repeats) != len(input_shape) or any(item <= 0 for item in repeats):
        raise ValueError("Tile repeats must be positive and match input rank")
    return [dim * repeat for dim, repeat in zip(input_shape, repeats)]


def _output_shape_for_resize(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    if "sizes" in attrs:
        sizes = _int_list(attrs.get("sizes"), input_shape)
        if len(sizes) != len(input_shape) or any(item <= 0 for item in sizes):
            raise ValueError("Resize sizes must be positive and match input rank")
        return sizes
    scales = attrs.get("scales")
    if scales is None:
        return input_shape
    parsed = _parse_literal(scales)
    if not isinstance(parsed, list) or len(parsed) != len(input_shape):
        raise ValueError("Resize scales must match input rank")
    return [max(1, int(round(dim * float(scale)))) for dim, scale in zip(input_shape, parsed)]


def _output_shape_for_conv(input_shape: list[int], attrs: dict[str, Any], weight_shape: list[int]) -> list[int]:
    if len(input_shape) != 4 or len(weight_shape) != 4:
        raise ValueError("Conv currently requires NCHW input and OIHW weights")
    pads = _int_list(attrs.get("pads"), [0, 0, 0, 0])
    strides = _int_list(attrs.get("strides"), [1, 1])
    dilations = _int_list(attrs.get("dilations"), [1, 1])
    out_channels, in_channels, kernel_h, kernel_w = weight_shape
    if input_shape[1] != in_channels:
        raise ValueError(f"Conv weight input channels {in_channels} do not match tensor channels {input_shape[1]}")
    out_h = ((input_shape[2] + pads[0] + pads[2] - dilations[0] * (kernel_h - 1) - 1) // strides[0]) + 1
    out_w = ((input_shape[3] + pads[1] + pads[3] - dilations[1] * (kernel_w - 1) - 1) // strides[1]) + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Conv output dimensions must be positive")
    return [input_shape[0], out_channels, out_h, out_w]


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


def _incoming_ids(node_id: str, op: str, incoming: dict[str, list[dict[str, Any]]]) -> list[str]:
    slot_order = INPUT_SLOT_ORDER.get(op, [])
    slot_index = {slot: index for index, slot in enumerate(slot_order)}
    seen_slots: set[str] = set()

    def sort_index(slot: str) -> int:
        if slot in slot_index:
            return slot_index[slot]
        if re.match(r"^in\d+$", slot):
            return int(slot[2:])
        return len(slot_order)

    for edge in incoming.get(node_id, []):
        slot = str(edge.get("targetHandle") or "").strip()
        if not slot:
            continue
        if slot not in slot_index and not re.match(r"^in\d+$", slot):
            raise ValueError(f"{op} node {node_id!r} has unknown input slot {slot!r}")
        if slot in seen_slots:
            raise ValueError(f"{op} node {node_id!r} has multiple edges for input slot {slot!r}")
        seen_slots.add(slot)
    edges = sorted(
        incoming.get(node_id, []),
        key=lambda edge: (
            sort_index(str(edge.get("targetHandle") or "").strip()),
            str(edge.get("source") or ""),
            str(edge.get("id") or ""),
        ),
    )
    return [str(edge["source"]) for edge in edges]


def _expect_inputs(op: str, node_id: str, ids: list[str]) -> None:
    required = {
        "Cast": 1,
        "Identity": 1,
        "Not": 1,
        "ReduceSum": 1,
        "ReduceMax": 1,
        "ReduceMin": 1,
        "ArgMax": 1,
        "Slice": 1,
        "Pad": 1,
        "Transpose": 1,
        "Tile": 1,
        "Resize": 1,
        "Conv": 1,
        "Equal": 2,
        "Greater": 2,
        "Less": 2,
        "GreaterOrEqual": 2,
        "LessOrEqual": 2,
        "And": 2,
        "Or": 2,
        "Xor": 2,
        "Add": 2,
        "Sub": 2,
        "Mul": 2,
        "Div": 2,
        "Mod": 2,
        "Where": 3,
        "Output": 1,
        "Relu": 1,
        "Abs": 1,
        "Neg": 1,
        "Floor": 1,
        "Clip": 1,
        "Sign": 1,
        "Sqrt": 1,
    }
    expected = required.get(op)
    if expected is not None and len(ids) != expected:
        raise ValueError(f"{op} node {node_id!r} requires {expected} input edge(s), got {len(ids)}")
    if op == "Concat" and len(ids) < 2:
        raise ValueError(f"Concat node {node_id!r} requires at least 2 input edge(s), got {len(ids)}")
    if op in {"Min", "Max", "Sum"} and len(ids) < 2:
        raise ValueError(f"{op} node {node_id!r} requires at least 2 input edge(s), got {len(ids)}")


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


def _coordinate_array(kind: str, shape: list[int]) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.float32)
    if kind == "row":
        arr[:, :, :, :] = np.arange(shape[-2], dtype=np.float32).reshape(1, 1, shape[-2], 1)
    else:
        arr[:, :, :, :] = np.arange(shape[-1], dtype=np.float32).reshape(1, 1, 1, shape[-1])
    return arr


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

        input_ids = _incoming_ids(node_id, op, incoming)

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

        if op in {"RowIndex", "ColIndex"}:
            shape = _shape(data, CANVAS_SHAPE)
            array = _coordinate_array("row" if op == "RowIndex" else "col", shape)
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

        same_shape_ops = {
            "Equal",
            "Greater",
            "Less",
            "GreaterOrEqual",
            "LessOrEqual",
            "And",
            "Or",
            "Xor",
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Mod",
            "Min",
            "Max",
            "Sum",
            "Where",
        }
        if op in same_shape_ops:
            for idx, input_shape in enumerate(input_shapes, start=1):
                if input_shape != input_shapes[0]:
                    raise ValueError(f"{op} node {node_id!r} input {idx} shape {input_shape} does not match {input_shapes[0]}")
            shape = input_shapes[0]
        if op in {"Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"}:
            out_type = TensorProto.BOOL
        elif op in {"Not", "And", "Or", "Xor"}:
            out_type = TensorProto.BOOL
        elif op == "Cast":
            out_type = _tensor_type_for_cast(attrs)
        elif op == "Where":
            if input_types[1] != input_types[2]:
                raise ValueError(f"Where node {node_id!r} true/false inputs must have matching tensor types")
            out_type = input_types[1]
        elif op in {"ReduceSum", "ReduceMax", "ReduceMin"}:
            axes = attrs.get("axes")
            shape = _output_shape_for_reduction(input_shapes[0], attrs)
            if op == "ReduceSum" and axes is not None:
                if isinstance(axes, int):
                    axes = [axes]
                axes_name = f"{node_id}_axes"
                initializers.append(helper.make_tensor(axes_name, TensorProto.INT64, [len(axes)], [int(axis) for axis in axes]))
                node_inputs = [inputs[0], axes_name]
        elif op == "Clip":
            if "min" in attrs or "max" in attrs:
                node_inputs = [inputs[0]]
                if "min" in attrs:
                    min_name = f"{node_id}_min"
                    initializers.append(helper.make_tensor(min_name, TensorProto.FLOAT, [], [float(attrs["min"])]))
                    node_inputs.append(min_name)
                elif "max" in attrs:
                    node_inputs.append("")
                if "max" in attrs:
                    max_name = f"{node_id}_max"
                    initializers.append(helper.make_tensor(max_name, TensorProto.FLOAT, [], [float(attrs["max"])]))
                    node_inputs.append(max_name)
        elif op == "ArgMax":
            shape = _output_shape_for_argmax(input_shapes[0], attrs)
            out_type = TensorProto.INT64
        elif op == "Slice":
            starts = _int_list(attrs.get("starts"), [0, 0, 0, 0])
            ends = _int_list(attrs.get("ends"), input_shapes[0])
            axes = _int_list(attrs.get("axes"), list(range(len(starts))))
            steps = _int_list(attrs.get("steps"), [1 for _ in starts])
            shape = _output_shape_for_slice(input_shapes[0], {"starts": starts, "ends": ends, "axes": axes, "steps": steps})
            for suffix, values in {"starts": starts, "ends": ends, "axes": axes, "steps": steps}.items():
                initializer_name = f"{node_id}_{suffix}"
                initializers.append(helper.make_tensor(initializer_name, TensorProto.INT64, [len(values)], [int(item) for item in values]))
                node_inputs.append(initializer_name)
        elif op == "Pad":
            pads = _int_list(attrs.get("pads"), [0 for _ in range(2 * len(input_shapes[0]))])
            shape = _output_shape_for_pad(input_shapes[0], {"pads": pads})
            pads_name = f"{node_id}_pads"
            value_name = f"{node_id}_constant_value"
            initializers.append(helper.make_tensor(pads_name, TensorProto.INT64, [len(pads)], [int(item) for item in pads]))
            initializers.append(helper.make_tensor(value_name, TensorProto.FLOAT, [], [float(attrs.get("value", 0))]))
            node_inputs = [inputs[0], pads_name, value_name]
        elif op == "Concat":
            shape = _output_shape_for_concat(input_shapes, attrs)
        elif op == "Transpose":
            shape = _output_shape_for_transpose(input_shapes[0], attrs)
        elif op == "Tile":
            repeats = _int_list(attrs.get("repeats"), [1 for _ in input_shapes[0]])
            shape = _output_shape_for_tile(input_shapes[0], {"repeats": repeats})
            repeats_name = f"{node_id}_repeats"
            initializers.append(helper.make_tensor(repeats_name, TensorProto.INT64, [len(repeats)], [int(item) for item in repeats]))
            node_inputs = [inputs[0], repeats_name]
        elif op == "Resize":
            shape = _output_shape_for_resize(input_shapes[0], attrs)
            roi_name = f"{node_id}_roi"
            scales_name = f"{node_id}_scales"
            sizes_name = f"{node_id}_sizes"
            initializers.append(helper.make_tensor(roi_name, TensorProto.FLOAT, [0], []))
            initializers.append(helper.make_tensor(scales_name, TensorProto.FLOAT, [0], []))
            initializers.append(helper.make_tensor(sizes_name, TensorProto.INT64, [len(shape)], [int(item) for item in shape]))
            node_inputs = [inputs[0], roi_name, scales_name, sizes_name]
        elif op == "Conv":
            weights = attrs.get("weights", attrs.get("kernel", [1]))
            weight_shape = _shape({"shape": attrs.get("weight_shape", attrs.get("kernel_shape", [1, input_shapes[0][1], 1, 1]))})
            weight_array = np.asarray(_parse_literal(weights), dtype=np.float32)
            if weight_array.size == 1 and int(np.prod(weight_shape)) != 1:
                weight_array = np.full(weight_shape, float(weight_array.reshape(-1)[0]), dtype=np.float32)
            else:
                weight_array = weight_array.reshape(weight_shape).astype(np.float32)
            weight_name = f"{node_id}_weights"
            initializers.append(numpy_helper.from_array(weight_array, name=weight_name))
            node_inputs = [inputs[0], weight_name]
            if "bias" in attrs:
                bias = np.asarray(_parse_literal(attrs["bias"]), dtype=np.float32).reshape([weight_shape[0]])
                bias_name = f"{node_id}_bias"
                initializers.append(numpy_helper.from_array(bias, name=bias_name))
                node_inputs.append(bias_name)
            shape = _output_shape_for_conv(input_shapes[0], attrs, weight_shape)

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


def _dim_value(dim: onnx.TensorShapeProto.Dimension) -> int | str | None:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return dim.dim_param
    return None


def _value_info_summary(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "elemType": int(tensor_type.elem_type),
        "shape": [_dim_value(dim) for dim in tensor_type.shape.dim],
    }


def _model_summary(model: onnx.ModelProto) -> dict[str, Any]:
    return {
        "inputs": [_value_info_summary(item) for item in model.graph.input],
        "outputs": [_value_info_summary(item) for item in model.graph.output],
    }


def _safe_id(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return text or fallback


def _shape_from_value_info(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    tensor_type = value_info.type.tensor_type
    return [_dim_value(dim) for dim in tensor_type.shape.dim]


def _json_ready(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, onnx.TensorProto):
        array = numpy_helper.to_array(value)
        return {
            "kind": "tensor",
            "dataType": int(value.data_type),
            "shape": list(array.shape),
            "size": int(array.size),
            "valuesPreview": array.reshape(-1)[:12].tolist(),
        }
    if isinstance(value, onnx.GraphProto):
        return {"kind": "graph", "name": value.name, "nodeCount": len(value.node)}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _onnx_attrs_for_gui(node: onnx.NodeProto) -> dict[str, Any]:
    return {attr.name: _json_ready(helper.get_attribute_value(attr)) for attr in node.attribute}


def _safe_source_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _best_onnx_path(task_id: str) -> Path:
    task_id = task_id.strip().lower()
    if not TASK_ID_RE.match(task_id):
        raise ValueError("taskId must match taskXXX, for example task010")
    candidates = [
        IMPORT_DIR / f"{task_id}.onnx",
        CLIENT_DIST / "best" / "onnx" / f"{task_id}.onnx",
        CLIENT_PUBLIC / "best" / "onnx" / f"{task_id}.onnx",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No best ONNX file found for {task_id}")


def onnx_to_gui_graph(task_id: str) -> dict[str, Any]:
    path = _best_onnx_path(task_id)
    model = onnx.load(path)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    producers: dict[str, str] = {}
    occupied_ids: set[str] = set()

    def unique_id(raw: str, fallback: str) -> str:
        base = _safe_id(raw, fallback)
        candidate = base
        suffix = 2
        while candidate in occupied_ids:
            candidate = f"{base}_{suffix}"
            suffix += 1
        occupied_ids.add(candidate)
        return candidate

    def add_node(raw_id: str, fallback: str, op_type: str, index: int, data: dict[str, Any]) -> str:
        node_id = unique_id(raw_id, fallback)
        nodes.append(
            {
                "id": node_id,
                "type": "op",
                "position": {"x": 70 + (index % 8) * 210, "y": 80 + (index // 8) * 130},
                "data": {"label": node_id, "opType": op_type, **data},
            }
        )
        return node_id

    index = 0
    initializer_names = {item.name for item in model.graph.initializer}
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        node_id = add_node(
            value_info.name,
            f"input_{index}",
            "Input",
            index,
            {
                "shape": ",".join(str(item) for item in _shape_from_value_info(value_info)),
                "sourceName": value_info.name,
                "inputSlots": [],
            },
        )
        producers[value_info.name] = node_id
        index += 1

    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        node_id = add_node(
            initializer.name,
            f"init_{index}",
            "Constant",
            index,
            {
                "shape": ",".join(str(item) for item in array.shape),
                "sourceName": initializer.name,
                "inputSlots": [],
                "attrs": {
                    "source": "onnx-initializer",
                    "dataType": int(initializer.data_type),
                    "size": int(array.size),
                    "valuesPreview": array.reshape(-1)[:12].tolist(),
                },
            },
        )
        producers[initializer.name] = node_id
        index += 1

    for node_index, onnx_node in enumerate(model.graph.node, start=1):
        slots = [f"in{slot_index}" for slot_index, name in enumerate(onnx_node.input) if name]
        node_id = add_node(
            onnx_node.name or f"{onnx_node.op_type}_{node_index}",
            f"onnx_{node_index}",
            onnx_node.op_type,
            index,
            {
                "sourceName": onnx_node.name,
                "inputSlots": slots,
                "attrs": _onnx_attrs_for_gui(onnx_node),
            },
        )
        index += 1
        visible_slot = 0
        for input_index, input_name in enumerate(onnx_node.input):
            if not input_name:
                continue
            source_id = producers.get(input_name)
            if source_id:
                slot = f"in{visible_slot}"
                edges.append(
                    {
                        "id": f"e_{len(edges) + 1}",
                        "source": source_id,
                        "target": node_id,
                        "targetHandle": slot,
                        "data": {"tensor": input_name, "inputIndex": input_index},
                    }
                )
            visible_slot += 1
        for output_name in onnx_node.output:
            if output_name:
                producers[output_name] = node_id

    for output_index, output in enumerate(model.graph.output, start=1):
        node_id = add_node(
            f"output_{output.name}",
            f"output_{output_index}",
            "Output",
            index,
            {
                "shape": ",".join(str(item) for item in _shape_from_value_info(output)),
                "sourceName": output.name,
                "inputSlots": ["input"],
            },
        )
        index += 1
        source_id = producers.get(output.name)
        if source_id:
            edges.append(
                {
                    "id": f"e_{len(edges) + 1}",
                    "source": source_id,
                    "target": node_id,
                    "targetHandle": "input",
                    "data": {"tensor": output.name},
                }
            )

    return {
        "projectName": f"best-{task_id}-onnx",
        "taskId": task_id,
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "source": _safe_source_label(path),
            "rawOnnx": True,
            "nodeCount": len(model.graph.node),
            "initializerCount": len(model.graph.initializer),
            "opTypes": sorted({node.op_type for node in model.graph.node}),
        },
    }


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


def _tensor_to_grid(tensor: np.ndarray, region: tuple[int, int]) -> list[list[int | float | bool]]:
    array = np.asarray(tensor)
    if array.ndim >= 4:
        array = array[0, 0]
    elif array.ndim == 3:
        array = array[0]
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    height, width = region
    if array.ndim == 2:
        array = array[:height, :width]
    rounded = np.rint(array)
    if np.all(np.isfinite(array)) and np.all(array == rounded):
        return rounded.astype(int).tolist()
    return array.tolist()


def validate_model(model: onnx.ModelProto, payload: ExportPayload) -> dict[str, str]:
    if not payload.trainingPairs:
        raise ValidationError("Phase 1 Strict Equivalence failed: no ARC training pairs were supplied")

    try:
        session = ort.InferenceSession(model.SerializeToString(), providers=_ort_providers())
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


@app.post("/api/compile")
def compile_onnx(payload: ExportPayload):
    try:
        model = compile_graph(payload)
        return {
            "status": "compiled",
            "taskId": _task_id(payload),
            "modelBytes": len(model.SerializeToString()),
            "nodeCount": len(payload.nodes),
            "edgeCount": len(payload.edges),
            "io": _model_summary(model),
        }
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})


@app.post("/api/run")
def run_onnx(payload: RunPayload):
    try:
        model = compile_graph(payload)
        source_grid = payload.inputGrid
        if source_grid is None:
            if not payload.trainingPairs:
                raise ValueError("Run requires inputGrid or at least one training pair")
            source_grid = payload.trainingPairs[0].get("input")
        if source_grid is None:
            raise ValueError("Run input grid is missing")
        session = ort.InferenceSession(model.SerializeToString(), providers=_ort_providers())
        actual, region = _run_session(session, source_grid)
        _assert_color_bounds("Run", actual)
        return {
            "status": "ran",
            "taskId": _task_id(payload),
            "shape": list(actual.shape),
            "grid": _tensor_to_grid(actual, region),
            "io": _model_summary(model),
        }
    except ValidationError as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})


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
            return FileResponse(
                str(artifact),
                media_type="application/octet-stream",
                filename=f"{task_id}.onnx",
                headers={"X-Validation": "passed"},
            )
        api = HfApi(token=token)
        _assert_hf_repo_matches_token(api, repo_id)
        remote_path = f"{task_id}.onnx"
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
            api.upload_file(path_or_fileobj=str(artifact), path_in_repo=remote_path, repo_id=repo_id, repo_type="model")
        except Exception as exc:
            return JSONResponse(
                status_code=502,
                content={
                    "status": "upload_failed",
                    "reason": f"Validation passed, but artifact push failed: {exc}",
                    "artifact": artifact.name,
                    "validation": validation,
                },
            )
        return {"status": "passed", "artifact": artifact.name, "repo": repo_id, "path": remote_path, "validation": validation}
    except ValidationError as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})


@app.get("/api/best-graph/{task_id}")
def best_graph(task_id: str):
    try:
        return onnx_to_gui_graph(task_id)
    except Exception as exc:
        return JSONResponse(status_code=404, content={"status": "failed", "reason": str(exc)})


_IMPORT_NAME_RE = re.compile(r"^(task\d{3})\.onnx$")


@app.post("/api/import")
async def import_onnx(files: list[UploadFile] = File(...)):
    saved: list[str] = []
    rejected: list[dict[str, str]] = []
    for upload in files:
        name = (upload.filename or "").strip().lower()
        match = _IMPORT_NAME_RE.match(name)
        if not match:
            rejected.append({"filename": upload.filename or "", "reason": "name must match taskNNN.onnx"})
            continue
        task_id = match.group(1)
        data = await upload.read()
        try:
            onnx.load_from_string(data)
        except Exception as exc:
            rejected.append({"filename": upload.filename, "reason": f"not a valid ONNX model: {exc}"})
            continue
        (IMPORT_DIR / f"{task_id}.onnx").write_bytes(data)
        saved.append(task_id)
    return {"status": "ok", "saved": saved, "rejected": rejected, "import_dir": str(IMPORT_DIR)}


@app.get("/api/import/list")
def import_list():
    items = []
    for path in sorted(IMPORT_DIR.glob("task*.onnx")):
        if _IMPORT_NAME_RE.match(path.name):
            items.append({"taskId": path.stem, "size": path.stat().st_size})
    return {"items": items, "import_dir": str(IMPORT_DIR)}


@app.delete("/api/import/{task_id}")
def import_delete(task_id: str):
    if not TASK_ID_RE.match(task_id):
        return JSONResponse(status_code=400, content={"status": "failed", "reason": "invalid taskId"})
    target = IMPORT_DIR / f"{task_id}.onnx"
    if not target.exists():
        return JSONResponse(status_code=404, content={"status": "failed", "reason": "not imported"})
    target.unlink()
    return {"status": "deleted", "taskId": task_id}


if CLIENT_DIST.exists():
    app.mount("/", StaticFiles(directory=str(CLIENT_DIST), html=True), name="static")

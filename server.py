import json
import math
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


def _ort_session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return opts


def _make_inference_session(model_bytes: bytes) -> ort.InferenceSession:
    return ort.InferenceSession(model_bytes, sess_options=_ort_session_options(), providers=_ort_providers())


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
    "Gather",
    "GatherND",
    "Squeeze",
    "Unsqueeze",
    "Reshape",
    "Expand",
    "OneHot",
    "MatMul",
    "MaxPool",
    "CumSum",
    "Flatten",
    "ScatterElements",
    "ConvTranspose",
    "GridSample",
    "GatherElements",
    "AveragePool",
    "Gemm",
    "RowIndex",
    "ColIndex",
    "Split",
    "TopK",
    "QLinearMatMul",
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
    "Gather": ["data", "indices"],
    "GatherND": ["data", "indices"],
    "Squeeze": ["input"],
    "Unsqueeze": ["input"],
    "Reshape": ["data", "shape"],
    "Expand": ["input", "shape"],
    "OneHot": ["indices", "depth", "values"],
    "MatMul": ["a", "b"],
    "MaxPool": ["input"],
    "CumSum": ["input", "axis"],
    "Flatten": ["input"],
    "ScatterElements": ["data", "indices", "updates"],
    "ConvTranspose": ["input"],
    "GridSample": ["input", "grid"],
    "GatherElements": ["data", "indices"],
    "AveragePool": ["input"],
    "Gemm": ["a", "b", "c"],
    "Split": ["input", "split"],
    "TopK": ["input", "k"],
    "QLinearMatMul": ["a", "a_scale", "a_zp", "b", "b_scale", "b_zp", "y_scale", "y_zp"],
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
    expectedOutput: Any | None = None
    traceIntermediates: bool = False


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
    raw = data.get("shape")
    if raw == "[]":
        return []
    if raw in ("", None):
        raw = default if default is not None else CANVAS_SHAPE
    raw = _parse_literal(raw)
    if isinstance(raw, int):
        raw = [raw]
    if isinstance(raw, str):
        raw = [int(part.strip()) for part in raw.replace("x", ",").split(",") if part.strip()]
    if not isinstance(raw, list) or not all(isinstance(item, int) and item > 0 for item in raw):
        raise ValueError("All tensors must have statically defined positive integer shapes")
    return raw


def _onnx_attrs(op: str, attrs: dict[str, Any]) -> dict[str, Any]:
    if op == "Cast":
        return {"to": int(attrs.get("to", TensorProto.FLOAT))}
    if op == "ArgMax":
        result: dict[str, Any] = {"axis": int(attrs.get("axis", 1)), "keepdims": int(attrs.get("keepdims", 1))}
        if "select_last_index" in attrs:
            result["select_last_index"] = int(attrs["select_last_index"])
        return result
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
        if "kernel_shape" in attrs:
            result["kernel_shape"] = [int(item) for item in attrs["kernel_shape"]]
        if "group" in attrs:
            result["group"] = int(attrs["group"])
        if "auto_pad" in attrs:
            result["auto_pad"] = str(attrs["auto_pad"])
        return result
    if op == "Mod":
        return {"fmod": int(attrs.get("fmod", 0))}
    if op == "Gather":
        return {"axis": int(attrs.get("axis", 0))}
    if op in {"Squeeze", "Unsqueeze", "Reshape", "Expand"}:
        return {}
    if op == "GatherND":
        return {"batch_dims": int(attrs.get("batch_dims", 0))}
    if op == "OneHot":
        return {"axis": int(attrs.get("axis", -1))}
    if op == "MatMul":
        return {}
    if op == "MaxPool":
        result: dict[str, Any] = {"kernel_shape": [int(item) for item in attrs.get("kernel_shape", [1, 1])]}
        if "pads" in attrs:
            result["pads"] = [int(item) for item in attrs["pads"]]
        if "strides" in attrs:
            result["strides"] = [int(item) for item in attrs["strides"]]
        if "dilations" in attrs:
            result["dilations"] = [int(item) for item in attrs["dilations"]]
        if "ceil_mode" in attrs:
            result["ceil_mode"] = int(attrs["ceil_mode"])
        return result
    if op == "CumSum":
        result = {}
        if "exclusive" in attrs:
            result["exclusive"] = int(attrs["exclusive"])
        if "reverse" in attrs:
            result["reverse"] = int(attrs["reverse"])
        return result
    if op == "Flatten":
        return {"axis": int(attrs.get("axis", 1))}
    if op == "ScatterElements":
        result: dict[str, Any] = {"axis": int(attrs.get("axis", 0))}
        if "reduction" in attrs:
            result["reduction"] = str(attrs["reduction"])
        return result
    if op == "ConvTranspose":
        result: dict[str, Any] = {}
        if "pads" in attrs:
            result["pads"] = [int(item) for item in attrs["pads"]]
        if "strides" in attrs:
            result["strides"] = [int(item) for item in attrs["strides"]]
        if "dilations" in attrs:
            result["dilations"] = [int(item) for item in attrs["dilations"]]
        if "kernel_shape" in attrs:
            result["kernel_shape"] = [int(item) for item in attrs["kernel_shape"]]
        if "group" in attrs:
            result["group"] = int(attrs["group"])
        if "output_padding" in attrs:
            result["output_padding"] = [int(item) for item in attrs["output_padding"]]
        if "output_shape" in attrs:
            result["output_shape"] = [int(item) for item in attrs["output_shape"]]
        if "auto_pad" in attrs:
            result["auto_pad"] = str(attrs["auto_pad"])
        return result
    if op == "GridSample":
        result: dict[str, Any] = {}
        if "align_corners" in attrs:
            result["align_corners"] = int(attrs["align_corners"])
        if "mode" in attrs:
            result["mode"] = str(attrs["mode"])
        if "padding_mode" in attrs:
            result["padding_mode"] = str(attrs["padding_mode"])
        return result
    if op == "GatherElements":
        return {"axis": int(attrs.get("axis", 0))}
    if op == "AveragePool":
        result: dict[str, Any] = {"kernel_shape": [int(item) for item in attrs.get("kernel_shape", [1, 1])]}
        if "pads" in attrs:
            result["pads"] = [int(item) for item in attrs["pads"]]
        if "strides" in attrs:
            result["strides"] = [int(item) for item in attrs["strides"]]
        if "ceil_mode" in attrs:
            result["ceil_mode"] = int(attrs["ceil_mode"])
        if "count_include_pad" in attrs:
            result["count_include_pad"] = int(attrs["count_include_pad"])
        if "auto_pad" in attrs:
            result["auto_pad"] = str(attrs["auto_pad"])
        return result
    if op == "Gemm":
        result: dict[str, Any] = {}
        if "alpha" in attrs:
            result["alpha"] = float(attrs["alpha"])
        if "beta" in attrs:
            result["beta"] = float(attrs["beta"])
        if "transA" in attrs:
            result["transA"] = int(attrs["transA"])
        if "transB" in attrs:
            result["transB"] = int(attrs["transB"])
        return result
    if op == "Split":
        result: dict[str, Any] = {"axis": int(attrs.get("axis", 0))}
        if "num_outputs" in attrs:
            result["num_outputs"] = int(attrs["num_outputs"])
        return result
    if op == "TopK":
        result: dict[str, Any] = {"axis": int(attrs.get("axis", -1))}
        if "largest" in attrs:
            result["largest"] = int(attrs["largest"])
        if "sorted" in attrs:
            result["sorted"] = int(attrs["sorted"])
        return result
    if op == "QLinearMatMul":
        return {}
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
    return [dim for idx, dim in enumerate(input_shape) if idx not in axes]


def _output_shape_for_argmax(input_shape: list[int], attrs: dict[str, Any]) -> list[int]:
    axis = int(attrs.get("axis", 1))
    keepdims = int(attrs.get("keepdims", 1))
    axis = (axis + len(input_shape)) % len(input_shape)
    if keepdims:
        return [1 if idx == axis else dim for idx, dim in enumerate(input_shape)]
    return [dim for idx, dim in enumerate(input_shape) if idx != axis]


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
        if step == 0:
            raise ValueError("Slice steps must be non-zero")
        dim = input_shape[axis]
        if step > 0:
            s = max(0, min(dim, start if start >= 0 else dim + start))
            e = max(0, min(dim, end if end >= 0 else dim + end))
            new_dim = max(0, (e - s + step - 1) // step)
        else:
            s = max(-1, min(dim - 1, start if start >= 0 else dim + start))
            e = max(-1, min(dim - 1, end if end >= 0 else dim + end))
            new_dim = max(0, (s - e + (-step) - 1) // (-step))
        shape[axis] = new_dim
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
    group = int(attrs.get("group", 1))
    out_channels, in_channels_per_group, kernel_h, kernel_w = weight_shape
    if input_shape[1] != in_channels_per_group * group:
        raise ValueError(
            f"Conv weight in_channels*group ({in_channels_per_group}*{group}={in_channels_per_group * group}) "
            f"do not match input channels {input_shape[1]}"
        )
    out_h = ((input_shape[2] + pads[0] + pads[2] - dilations[0] * (kernel_h - 1) - 1) // strides[0]) + 1
    out_w = ((input_shape[3] + pads[1] + pads[3] - dilations[1] * (kernel_w - 1) - 1) // strides[1]) + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Conv output dimensions must be positive")
    return [input_shape[0], out_channels, out_h, out_w]


def _tensor_type_for_cast(attrs: dict[str, Any]) -> int:
    return int(attrs.get("to", TensorProto.FLOAT))


def _np_dtype_for_tensor_proto(dtype: int) -> np.dtype:
    return np.dtype(helper.tensor_dtype_to_np_dtype(dtype))


def _broadcast_shapes(shapes: list[list[int]]) -> list[int]:
    if not shapes:
        return []
    max_rank = max(len(shape) for shape in shapes)
    padded = [[1] * (max_rank - len(shape)) + list(shape) for shape in shapes]
    result = []
    for axis_index, dims in enumerate(zip(*padded)):
        non_one = {dim for dim in dims if dim != 1}
        if len(non_one) > 1:
            raise ValueError(
                f"shapes {shapes} cannot be broadcast on axis {axis_index} (sizes {sorted(non_one)})"
            )
        result.append(max(dims))
    return result


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


def _incoming_ids(node_id: str, op: str, incoming: dict[str, list[dict[str, Any]]]) -> list[tuple[str, int]]:
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

    def _out_idx(edge: dict[str, Any]) -> int:
        handle = str(edge.get("sourceHandle") or "").strip()
        match = re.match(r"^out(\d+)$", handle)
        return int(match.group(1)) if match else 0

    return [(str(edge["source"]), _out_idx(edge)) for edge in edges]


def _expect_inputs(op: str, node_id: str, ids: list[str]) -> None:
    required = {
        "Cast": 1,
        "Identity": 1,
        "Not": 1,
        "ArgMax": 1,
        "Transpose": 1,
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
        "Sign": 1,
        "Sqrt": 1,
        "Gather": 2,
    }
    flexible = {
        "Tile": (1, 2),
        "Slice": (1, 5),
        "Pad": (1, 3),
        "ReduceSum": (1, 2),
        "ReduceMax": (1, 2),
        "ReduceMin": (1, 2),
        "Squeeze": (1, 2),
        "Unsqueeze": (1, 2),
        "Reshape": (2, 2),
        "Expand": (2, 2),
        "Conv": (1, 3),
        "GatherND": (2, 2),
        "Clip": (1, 3),
        "OneHot": (3, 3),
        "MatMul": (2, 2),
        "MaxPool": (1, 1),
        "CumSum": (2, 2),
        "Flatten": (1, 1),
        "Resize": (1, 4),
        "ScatterElements": (3, 3),
        "ConvTranspose": (1, 3),
        "GridSample": (2, 2),
        "GatherElements": (2, 2),
        "AveragePool": (1, 1),
        "Gemm": (2, 3),
        "Split": (1, 2),
        "TopK": (2, 2),
        "QLinearMatMul": (8, 8),
    }
    expected = required.get(op)
    if expected is not None and len(ids) != expected:
        raise ValueError(f"{op} node {node_id!r} requires {expected} input edge(s), got {len(ids)}")
    if op in flexible:
        low, high = flexible[op]
        if not (low <= len(ids) <= high):
            raise ValueError(f"{op} node {node_id!r} requires {low}–{high} input edge(s), got {len(ids)}")
    if op == "Concat" and len(ids) < 2:
        raise ValueError(f"Concat node {node_id!r} requires at least 2 input edge(s), got {len(ids)}")
    if op in {"Min", "Max", "Sum"} and len(ids) < 2:
        raise ValueError(f"{op} node {node_id!r} requires at least 2 input edge(s), got {len(ids)}")


def _constant_array(data: dict[str, Any], shape: list[int]) -> np.ndarray:
    dtype_int = int(data.get("dataType", TensorProto.FLOAT))
    np_dtype = _np_dtype_for_tensor_proto(dtype_int)
    values = data.get("values", data.get("value"))
    if values is None:
        return np.zeros(shape, dtype=np_dtype)
    array = np.asarray(_parse_literal(values))
    if array.size == 1 and int(np.prod(shape)) != 1:
        return np.full(shape, array.reshape(-1)[0], dtype=np_dtype)
    try:
        return array.reshape(shape).astype(np_dtype)
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
    tensor_name: dict[tuple[str, int], str] = {}
    tensor_shape: dict[tuple[str, int], list[int]] = {}
    tensor_type: dict[tuple[str, int], int] = {}
    constant_values: dict[str, np.ndarray] = {}

    for node in sorted_nodes:
        node_id = _node_id(node)
        data = node.get("data") or {}
        op = _op_type(node)
        attrs = _raw_attrs(data)
        output_name = f"{node_id}_out"

        if op == "Input":
            shape = _shape(data, CANVAS_SHAPE)
            tensor_name[(node_id, 0)] = output_name
            tensor_shape[(node_id, 0)] = shape
            tensor_type[(node_id, 0)] = TensorProto.FLOAT
            graph_inputs.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape))
            continue

        input_refs = _incoming_ids(node_id, op, incoming)
        input_ids = [ref[0] for ref in input_refs]

        if op == "Output":
            _expect_inputs(op, node_id, input_ids)
            ref = input_refs[0]
            source_name = tensor_name[ref]
            graph_outputs.append(helper.make_tensor_value_info(source_name, tensor_type[ref], tensor_shape[ref]))
            continue

        if op == "Constant":
            inferred_default = CANVAS_SHAPE
            if not data.get("shape"):
                raw_values = data.get("values", data.get("value"))
                if raw_values is not None:
                    try:
                        probe = np.asarray(_parse_literal(raw_values))
                        if probe.size == 1:
                            inferred_default = [1]
                        elif probe.ndim >= 1 and list(probe.shape) != CANVAS_SHAPE:
                            inferred_default = list(probe.shape)
                    except Exception:
                        pass
            shape = _shape(data, inferred_default)
            array = _constant_array(data, shape)
            tensor = numpy_helper.from_array(array, name=f"{node_id}_value")
            dtype_int = int(tensor.data_type)
            tensor_name[(node_id, 0)] = output_name
            tensor_shape[(node_id, 0)] = shape
            tensor_type[(node_id, 0)] = dtype_int
            constant_values[node_id] = array
            onnx_nodes.append(helper.make_node("Constant", inputs=[], outputs=[output_name], name=node_id, value=tensor))
            value_infos.append(helper.make_tensor_value_info(output_name, dtype_int, shape))
            continue

        if op in {"RowIndex", "ColIndex"}:
            shape = _shape(data, CANVAS_SHAPE)
            array = _coordinate_array("row" if op == "RowIndex" else "col", shape)
            tensor = numpy_helper.from_array(array, name=f"{node_id}_value")
            tensor_name[(node_id, 0)] = output_name
            tensor_shape[(node_id, 0)] = shape
            tensor_type[(node_id, 0)] = TensorProto.FLOAT
            onnx_nodes.append(helper.make_node("Constant", inputs=[], outputs=[output_name], name=node_id, value=tensor))
            value_infos.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape))
            continue

        _expect_inputs(op, node_id, input_ids)
        inputs = [tensor_name[ref] for ref in input_refs]
        input_shapes = [tensor_shape[ref] for ref in input_refs]
        input_types = [tensor_type[ref] for ref in input_refs]
        shape = _shape(data, input_shapes[0])
        out_type = input_types[0]
        node_inputs = inputs
        node_attrs_override: dict[str, Any] | None = None
        extra_outputs: list[tuple[str, list[int], int]] = []

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
            try:
                shape = _broadcast_shapes(input_shapes)
            except ValueError as exc:
                raise ValueError(f"{op} node {node_id!r}: {exc}") from exc
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
            if len(input_ids) >= 2:
                if input_ids[1] not in constant_values:
                    raise ValueError(f"{op} node {node_id!r} expects its axes input to be a Constant")
                axes_values = [int(item) for item in constant_values[input_ids[1]].reshape(-1).tolist()]
                shape = _output_shape_for_reduction(input_shapes[0], {**attrs, "axes": axes_values})
                node_inputs = [inputs[0], inputs[1]]
                node_attrs_override = {"keepdims": int(attrs.get("keepdims", 1))}
            else:
                axes = attrs.get("axes")
                if op == "ReduceSum" and axes is not None:
                    if isinstance(axes, int):
                        axes = [axes]
                    axes_name = f"{node_id}_axes"
                    initializers.append(helper.make_tensor(axes_name, TensorProto.INT64, [len(axes)], [int(axis) for axis in axes]))
                    node_inputs = [inputs[0], axes_name]
                shape = _output_shape_for_reduction(input_shapes[0], attrs)
        elif op == "Clip":
            if len(input_ids) >= 2:
                node_inputs = list(inputs)
            elif "min" in attrs or "max" in attrs:
                node_inputs = [inputs[0]]
                in_dtype = input_types[0]
                np_dtype = _np_dtype_for_tensor_proto(in_dtype)
                if "min" in attrs:
                    min_name = f"{node_id}_min"
                    min_arr = np.array(attrs["min"], dtype=np_dtype)
                    initializers.append(numpy_helper.from_array(min_arr, name=min_name))
                    node_inputs.append(min_name)
                elif "max" in attrs:
                    node_inputs.append("")
                if "max" in attrs:
                    max_name = f"{node_id}_max"
                    max_arr = np.array(attrs["max"], dtype=np_dtype)
                    initializers.append(numpy_helper.from_array(max_arr, name=max_name))
                    node_inputs.append(max_name)
        elif op == "ArgMax":
            shape = _output_shape_for_argmax(input_shapes[0], attrs)
            out_type = TensorProto.INT64
        elif op == "Slice":
            if len(input_ids) == 1:
                starts = _int_list(attrs.get("starts"), [0, 0, 0, 0])
                ends = _int_list(attrs.get("ends"), input_shapes[0])
                axes = _int_list(attrs.get("axes"), list(range(len(starts))))
                steps = _int_list(attrs.get("steps"), [1 for _ in starts])
                shape = _output_shape_for_slice(input_shapes[0], {"starts": starts, "ends": ends, "axes": axes, "steps": steps})
                node_inputs = [inputs[0]]
                for suffix, values in {"starts": starts, "ends": ends, "axes": axes, "steps": steps}.items():
                    initializer_name = f"{node_id}_{suffix}"
                    initializers.append(helper.make_tensor(initializer_name, TensorProto.INT64, [len(values)], [int(item) for item in values]))
                    node_inputs.append(initializer_name)
            else:
                dyn = any(
                    idx < len(input_ids) and input_ids[idx] not in constant_values
                    for idx in range(1, min(5, len(input_ids)))
                )
                if dyn:
                    shape = list(input_shapes[0])
                else:
                    starts = [int(item) for item in constant_values[input_ids[1]].reshape(-1).tolist()]
                    ends = [int(item) for item in constant_values[input_ids[2]].reshape(-1).tolist()]
                    axes = (
                        [int(item) for item in constant_values[input_ids[3]].reshape(-1).tolist()]
                        if len(input_ids) >= 4
                        else list(range(len(starts)))
                    )
                    steps = (
                        [int(item) for item in constant_values[input_ids[4]].reshape(-1).tolist()]
                        if len(input_ids) >= 5
                        else [1 for _ in starts]
                    )
                    shape = _output_shape_for_slice(input_shapes[0], {"starts": starts, "ends": ends, "axes": axes, "steps": steps})
                node_inputs = list(inputs)
        elif op == "Pad":
            if len(input_ids) == 1:
                pads = _int_list(attrs.get("pads"), [0 for _ in range(2 * len(input_shapes[0]))])
                shape = _output_shape_for_pad(input_shapes[0], {"pads": pads})
                pads_name = f"{node_id}_pads"
                value_name = f"{node_id}_constant_value"
                in_dtype = input_types[0]
                initializers.append(helper.make_tensor(pads_name, TensorProto.INT64, [len(pads)], [int(item) for item in pads]))
                np_value = np.array(attrs.get("value", 0), dtype=_np_dtype_for_tensor_proto(in_dtype))
                initializers.append(numpy_helper.from_array(np_value, name=value_name))
                node_inputs = [inputs[0], pads_name, value_name]
            else:
                if input_ids[1] not in constant_values:
                    raise ValueError(f"Pad node {node_id!r} expects its pads input to be a Constant")
                pads = [int(item) for item in constant_values[input_ids[1]].reshape(-1).tolist()]
                shape = _output_shape_for_pad(input_shapes[0], {"pads": pads})
                node_inputs = list(inputs)
        elif op == "Concat":
            shape = _output_shape_for_concat(input_shapes, attrs)
        elif op == "Transpose":
            shape = _output_shape_for_transpose(input_shapes[0], attrs)
        elif op == "Tile":
            if len(input_ids) == 1:
                repeats = _int_list(attrs.get("repeats"), [1 for _ in input_shapes[0]])
                shape = _output_shape_for_tile(input_shapes[0], {"repeats": repeats})
                repeats_name = f"{node_id}_repeats"
                initializers.append(helper.make_tensor(repeats_name, TensorProto.INT64, [len(repeats)], [int(item) for item in repeats]))
                node_inputs = [inputs[0], repeats_name]
            else:
                if input_ids[1] not in constant_values:
                    raise ValueError(f"Tile node {node_id!r} expects its repeats input to be a Constant")
                repeats = [int(item) for item in constant_values[input_ids[1]].reshape(-1).tolist()]
                shape = _output_shape_for_tile(input_shapes[0], {"repeats": repeats})
                node_inputs = list(inputs)
        elif op == "Resize":
            if len(input_ids) >= 2:
                sizes = None
                scales = None
                if len(input_ids) >= 4 and input_ids[3] in constant_values:
                    sizes = [int(item) for item in constant_values[input_ids[3]].reshape(-1).tolist()]
                if scales is None and len(input_ids) >= 3 and input_ids[2] in constant_values:
                    raw = constant_values[input_ids[2]].reshape(-1).tolist()
                    if raw:
                        scales = [float(item) for item in raw]
                if len(input_ids) == 2 and input_ids[1] in constant_values:
                    arr = constant_values[input_ids[1]]
                    raw = arr.reshape(-1).tolist()
                    if raw:
                        if np.issubdtype(arr.dtype, np.floating):
                            scales = [float(item) for item in raw]
                        else:
                            sizes = [int(item) for item in raw]
                if sizes is not None and any(s > 0 for s in sizes):
                    shape = sizes
                elif scales is not None and len(scales) == len(input_shapes[0]):
                    shape = [max(1, int(round(d * s))) for d, s in zip(input_shapes[0], scales)]
                else:
                    shape = _output_shape_for_resize(input_shapes[0], attrs)
                if len(input_ids) == 2:
                    roi_name = f"{node_id}_roi"
                    initializers.append(helper.make_tensor(roi_name, TensorProto.FLOAT, [0], []))
                    if scales is not None:
                        node_inputs = [inputs[0], roi_name, inputs[1]]
                    else:
                        scales_name = f"{node_id}_scales"
                        initializers.append(helper.make_tensor(scales_name, TensorProto.FLOAT, [0], []))
                        node_inputs = [inputs[0], roi_name, scales_name, inputs[1]]
                else:
                    node_inputs = list(inputs)
            else:
                shape = _output_shape_for_resize(input_shapes[0], attrs)
                roi_name = f"{node_id}_roi"
                scales_name = f"{node_id}_scales"
                sizes_name = f"{node_id}_sizes"
                initializers.append(helper.make_tensor(roi_name, TensorProto.FLOAT, [0], []))
                initializers.append(helper.make_tensor(scales_name, TensorProto.FLOAT, [0], []))
                initializers.append(helper.make_tensor(sizes_name, TensorProto.INT64, [len(shape)], [int(item) for item in shape]))
                node_inputs = [inputs[0], roi_name, scales_name, sizes_name]
        elif op == "Conv":
            if len(input_ids) >= 2:
                if input_ids[1] in constant_values:
                    weight_shape = list(constant_values[input_ids[1]].shape)
                else:
                    weight_shape = list(input_shapes[1])
                node_inputs = list(inputs)
            else:
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
        elif op == "Squeeze":
            in_shape = list(input_shapes[0])
            if len(input_ids) >= 2:
                if input_ids[1] not in constant_values:
                    raise ValueError(f"Squeeze node {node_id!r} expects axes input to be a Constant")
                axes_list = [int(item) for item in constant_values[input_ids[1]].reshape(-1).tolist()]
                node_inputs = [inputs[0], inputs[1]]
            else:
                raw_axes = attrs.get("axes")
                if isinstance(raw_axes, int):
                    axes_list = [raw_axes]
                elif isinstance(raw_axes, list):
                    axes_list = [int(item) for item in raw_axes]
                else:
                    axes_list = []
                if axes_list:
                    axes_name = f"{node_id}_axes"
                    initializers.append(helper.make_tensor(axes_name, TensorProto.INT64, [len(axes_list)], axes_list))
                    node_inputs = [inputs[0], axes_name]
            if axes_list:
                normalized = sorted({(a + len(in_shape)) % len(in_shape) for a in axes_list})
                shape = [d for i, d in enumerate(in_shape) if i not in normalized]
            else:
                shape = [d for d in in_shape if d != 1]
        elif op == "Unsqueeze":
            in_shape = list(input_shapes[0])
            if len(input_ids) >= 2:
                if input_ids[1] not in constant_values:
                    raise ValueError(f"Unsqueeze node {node_id!r} expects axes input to be a Constant")
                axes_list = [int(item) for item in constant_values[input_ids[1]].reshape(-1).tolist()]
                node_inputs = [inputs[0], inputs[1]]
            else:
                raw_axes = attrs.get("axes")
                if isinstance(raw_axes, int):
                    axes_list = [raw_axes]
                elif isinstance(raw_axes, list):
                    axes_list = [int(item) for item in raw_axes]
                else:
                    axes_list = []
                if not axes_list:
                    raise ValueError(f"Unsqueeze node {node_id!r} requires axes")
                axes_name = f"{node_id}_axes"
                initializers.append(helper.make_tensor(axes_name, TensorProto.INT64, [len(axes_list)], axes_list))
                node_inputs = [inputs[0], axes_name]
            out_rank = len(in_shape) + len(axes_list)
            normalized = sorted({(a + out_rank) % out_rank for a in axes_list})
            shape = list(in_shape)
            for ax in normalized:
                shape.insert(ax, 1)
        elif op == "Reshape":
            if input_ids[1] not in constant_values:
                raise ValueError(f"Reshape node {node_id!r} expects shape input to be a Constant")
            target = [int(item) for item in constant_values[input_ids[1]].reshape(-1).tolist()]
            in_shape = list(input_shapes[0])
            in_size = 1
            for d in in_shape:
                in_size *= d
            resolved = []
            minus_one_index = -1
            for i, dim in enumerate(target):
                if dim == 0:
                    resolved.append(in_shape[i] if i < len(in_shape) else 1)
                elif dim == -1:
                    if minus_one_index >= 0:
                        raise ValueError(f"Reshape node {node_id!r} has more than one -1 dim")
                    resolved.append(-1)
                    minus_one_index = i
                else:
                    resolved.append(int(dim))
            if minus_one_index >= 0:
                known = 1
                for dim in resolved:
                    if dim != -1:
                        known *= dim
                if known == 0:
                    raise ValueError(f"Reshape node {node_id!r} cannot infer -1 (other dims are 0)")
                resolved[minus_one_index] = in_size // known
            shape = resolved
            node_inputs = [inputs[0], inputs[1]]
        elif op == "Expand":
            if input_ids[1] not in constant_values:
                raise ValueError(f"Expand node {node_id!r} expects shape input to be a Constant")
            target = [int(item) for item in constant_values[input_ids[1]].reshape(-1).tolist()]
            try:
                shape = _broadcast_shapes([list(input_shapes[0]), target])
            except ValueError as exc:
                raise ValueError(f"Expand node {node_id!r}: {exc}") from exc
            node_inputs = [inputs[0], inputs[1]]
        elif op == "GatherND":
            batch_dims = int(attrs.get("batch_dims", 0))
            data_shape = list(input_shapes[0])
            indices_shape = list(input_shapes[1])
            last = indices_shape[-1] if indices_shape else 0
            shape = indices_shape[:-1] + data_shape[batch_dims + last:]
            out_type = input_types[0]
        elif op == "Gather":
            axis = int(attrs.get("axis", 0))
            data_shape = list(input_shapes[0])
            indices_shape = list(input_shapes[1])
            actual_axis = (axis + len(data_shape)) % len(data_shape)
            shape = data_shape[:actual_axis] + indices_shape + data_shape[actual_axis + 1:]
            out_type = input_types[0]
        elif op == "OneHot":
            if input_ids[1] not in constant_values:
                raise ValueError(f"OneHot node {node_id!r} expects depth input to be a Constant")
            depth_arr = constant_values[input_ids[1]].reshape(-1).tolist()
            depth = int(depth_arr[0])
            axis = int(attrs.get("axis", -1))
            in_shape = list(input_shapes[0])
            insert_at = (axis + len(in_shape) + 1) % (len(in_shape) + 1)
            shape = in_shape[:insert_at] + [depth] + in_shape[insert_at:]
            out_type = input_types[2]
            node_inputs = list(inputs)
        elif op == "MatMul":
            a_shape = list(input_shapes[0])
            b_shape = list(input_shapes[1])
            if len(a_shape) >= 2 and len(b_shape) >= 2:
                m, k1 = a_shape[-2], a_shape[-1]
                k2, n = b_shape[-2], b_shape[-1]
                if k1 != k2:
                    raise ValueError(f"MatMul node {node_id!r} inner dims mismatch: {k1} vs {k2}")
                a_batch = a_shape[:-2] or [1]
                b_batch = b_shape[:-2] or [1]
                batch_shape = _broadcast_shapes([a_batch, b_batch])
                shape = (batch_shape if (a_shape[:-2] or b_shape[:-2]) else []) + [m, n]
            elif len(a_shape) == 1 and len(b_shape) == 1:
                shape = [1]
            elif len(a_shape) == 1:
                shape = list(b_shape[:-2]) + [b_shape[-1]]
            elif len(b_shape) == 1:
                shape = list(a_shape[:-1])
            else:
                raise ValueError(f"MatMul node {node_id!r}: cannot infer shape for {a_shape} x {b_shape}")
        elif op == "MaxPool":
            kernel = _int_list(attrs.get("kernel_shape"), [1, 1])
            pads = _int_list(attrs.get("pads"), [0] * (2 * len(kernel)))
            strides = _int_list(attrs.get("strides"), [1 for _ in kernel])
            dilations = _int_list(attrs.get("dilations"), [1 for _ in kernel])
            in_shape = list(input_shapes[0])
            if len(in_shape) != 4 or len(kernel) != 2:
                raise ValueError(f"MaxPool node {node_id!r} requires NCHW input and 2D kernel")
            ceil_mode = int(attrs.get("ceil_mode", 0))
            def _pool_dim(d, p_lo, p_hi, k, s, dil):
                num = d + p_lo + p_hi - dil * (k - 1) - 1
                if ceil_mode:
                    return -(-num // s) + 1
                return num // s + 1
            out_h = _pool_dim(in_shape[2], pads[0], pads[2], kernel[0], strides[0], dilations[0])
            out_w = _pool_dim(in_shape[3], pads[1], pads[3], kernel[1], strides[1], dilations[1])
            shape = [in_shape[0], in_shape[1], out_h, out_w]
        elif op == "CumSum":
            shape = list(input_shapes[0])
            node_inputs = list(inputs)
        elif op == "Flatten":
            axis = int(attrs.get("axis", 1))
            in_shape = list(input_shapes[0])
            actual_axis = (axis + len(in_shape)) % len(in_shape) if in_shape else 0
            pre = 1
            for d in in_shape[:actual_axis]:
                pre *= d
            post = 1
            for d in in_shape[actual_axis:]:
                post *= d
            shape = [pre, post]
        elif op == "ScatterElements":
            shape = list(input_shapes[0])
            node_inputs = list(inputs)
        elif op == "GridSample":
            in_shape = list(input_shapes[0])
            grid_shape = list(input_shapes[1])
            if len(in_shape) == 4 and len(grid_shape) == 4:
                shape = [in_shape[0], in_shape[1], grid_shape[1], grid_shape[2]]
            elif len(in_shape) == 5 and len(grid_shape) == 5:
                shape = [in_shape[0], in_shape[1], grid_shape[1], grid_shape[2], grid_shape[3]]
            else:
                raise ValueError(f"GridSample node {node_id!r}: unsupported ranks {len(in_shape)}/{len(grid_shape)}")
            node_inputs = list(inputs)
        elif op == "GatherElements":
            shape = list(input_shapes[1])
            node_inputs = list(inputs)
        elif op == "AveragePool":
            kernel = _int_list(attrs.get("kernel_shape"), [1, 1])
            pads = _int_list(attrs.get("pads"), [0] * (2 * len(kernel)))
            strides = _int_list(attrs.get("strides"), [1 for _ in kernel])
            in_shape = list(input_shapes[0])
            if len(in_shape) != 4 or len(kernel) != 2:
                raise ValueError(f"AveragePool node {node_id!r} requires NCHW input and 2D kernel")
            ceil_mode = int(attrs.get("ceil_mode", 0))
            def _avgpool_dim(d, p_lo, p_hi, k, s):
                num = d + p_lo + p_hi - k
                if ceil_mode:
                    return -(-num // s) + 1
                return num // s + 1
            out_h = _avgpool_dim(in_shape[2], pads[0], pads[2], kernel[0], strides[0])
            out_w = _avgpool_dim(in_shape[3], pads[1], pads[3], kernel[1], strides[1])
            shape = [in_shape[0], in_shape[1], out_h, out_w]
        elif op == "Gemm":
            a_shape = list(input_shapes[0])
            b_shape = list(input_shapes[1])
            trans_a = int(attrs.get("transA", 0))
            trans_b = int(attrs.get("transB", 0))
            m_dim = a_shape[1] if trans_a else a_shape[0]
            n_dim = b_shape[0] if trans_b else b_shape[1]
            shape = [m_dim, n_dim]
            node_inputs = list(inputs)
        elif op == "ConvTranspose":
            in_shape = list(input_shapes[0])
            if len(input_ids) >= 2:
                if input_ids[1] in constant_values:
                    weight_shape = list(constant_values[input_ids[1]].shape)
                else:
                    weight_shape = list(input_shapes[1])
                node_inputs = list(inputs)
            else:
                raise ValueError(f"ConvTranspose node {node_id!r} requires a weight input")
            group = int(attrs.get("group", 1))
            kernel = _int_list(attrs.get("kernel_shape"), list(weight_shape[2:]))
            strides = _int_list(attrs.get("strides"), [1 for _ in kernel])
            pads = _int_list(attrs.get("pads"), [0] * (2 * len(kernel)))
            dilations = _int_list(attrs.get("dilations"), [1 for _ in kernel])
            output_padding = _int_list(attrs.get("output_padding"), [0 for _ in kernel])
            out_channels = weight_shape[1] * group
            spatial = []
            for i, k in enumerate(kernel):
                in_d = in_shape[2 + i]
                p_lo = pads[i]
                p_hi = pads[i + len(kernel)]
                out_d = strides[i] * (in_d - 1) + output_padding[i] + ((k - 1) * dilations[i] + 1) - p_lo - p_hi
                spatial.append(out_d)
            shape = [in_shape[0], out_channels] + spatial
        elif op == "Split":
            in_shape = list(input_shapes[0])
            axis = int(attrs.get("axis", 0))
            actual_axis = (axis + len(in_shape)) % len(in_shape) if in_shape else 0
            split_sizes: list[int] | None = None
            if len(input_ids) >= 2:
                if input_ids[1] in constant_values:
                    split_sizes = [int(item) for item in constant_values[input_ids[1]].reshape(-1).tolist()]
                else:
                    raise ValueError(f"Split node {node_id!r} expects its split input to be a Constant")
                node_inputs = list(inputs)
            elif "split" in attrs and attrs["split"] is not None:
                raw_split = attrs["split"]
                split_sizes = [int(raw_split)] if isinstance(raw_split, int) else [int(item) for item in raw_split]
                split_name = f"{node_id}_split"
                initializers.append(helper.make_tensor(split_name, TensorProto.INT64, [len(split_sizes)], split_sizes))
                node_inputs = [inputs[0], split_name]
            num_outputs_hint = data.get("outputCount")
            if num_outputs_hint is None and "num_outputs" in attrs:
                num_outputs_hint = attrs["num_outputs"]
            if split_sizes is None:
                if num_outputs_hint is None:
                    raise ValueError(f"Split node {node_id!r} requires split sizes or outputCount")
                num_outputs = int(num_outputs_hint)
                axis_dim = in_shape[actual_axis]
                base = axis_dim // num_outputs
                remainder = axis_dim - base * num_outputs
                split_sizes = [base + (1 if i < remainder else 0) for i in range(num_outputs)]
            num_outputs = len(split_sizes)
            shape = list(in_shape)
            shape[actual_axis] = split_sizes[0]
            for i in range(1, num_outputs):
                extra_shape = list(in_shape)
                extra_shape[actual_axis] = split_sizes[i]
                extra_outputs.append((f"{node_id}_out_{i}", extra_shape, input_types[0]))
            node_attrs_override = {"axis": axis}
        elif op == "TopK":
            in_shape = list(input_shapes[0])
            axis = int(attrs.get("axis", -1))
            actual_axis = (axis + len(in_shape)) % len(in_shape) if in_shape else 0
            if len(input_ids) < 2 or input_ids[1] not in constant_values:
                raise ValueError(f"TopK node {node_id!r} expects its K input to be a Constant")
            k_arr = constant_values[input_ids[1]].reshape(-1)
            k_val = int(k_arr[0])
            shape = list(in_shape)
            shape[actual_axis] = k_val
            extra_outputs.append((f"{node_id}_out_1", list(shape), TensorProto.INT64))
            node_inputs = list(inputs)
            override: dict[str, Any] = {"axis": axis}
            if "largest" in attrs:
                override["largest"] = int(attrs["largest"])
            if "sorted" in attrs:
                override["sorted"] = int(attrs["sorted"])
            node_attrs_override = override
        elif op == "QLinearMatMul":
            a_shape = list(input_shapes[0])
            b_shape = list(input_shapes[3])
            if len(a_shape) >= 2 and len(b_shape) >= 2:
                m, k1 = a_shape[-2], a_shape[-1]
                k2, n = b_shape[-2], b_shape[-1]
                if k1 != k2:
                    raise ValueError(f"QLinearMatMul node {node_id!r} inner dims mismatch: {k1} vs {k2}")
                a_batch = a_shape[:-2] or [1]
                b_batch = b_shape[:-2] or [1]
                batch_shape = _broadcast_shapes([a_batch, b_batch])
                shape = (batch_shape if (a_shape[:-2] or b_shape[:-2]) else []) + [m, n]
            elif len(a_shape) == 1 and len(b_shape) == 1:
                shape = [1]
            elif len(a_shape) == 1:
                shape = list(b_shape[:-2]) + [b_shape[-1]]
            elif len(b_shape) == 1:
                shape = list(a_shape[:-1])
            else:
                raise ValueError(f"QLinearMatMul node {node_id!r}: cannot infer shape for {a_shape} x {b_shape}")
            out_type = input_types[0]
            node_inputs = list(inputs)

        tensor_name[(node_id, 0)] = output_name
        tensor_shape[(node_id, 0)] = shape
        tensor_type[(node_id, 0)] = out_type
        output_names = [output_name]
        for extra_idx, (extra_name, extra_shape, extra_type) in enumerate(extra_outputs, start=1):
            tensor_name[(node_id, extra_idx)] = extra_name
            tensor_shape[(node_id, extra_idx)] = extra_shape
            tensor_type[(node_id, extra_idx)] = extra_type
            output_names.append(extra_name)
            value_infos.append(helper.make_tensor_value_info(extra_name, extra_type, extra_shape))
        final_attrs = node_attrs_override if node_attrs_override is not None else _onnx_attrs(op, attrs)
        onnx_nodes.append(helper.make_node(op, inputs=node_inputs, outputs=output_names, name=node_id, **final_attrs))
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


_DTYPE_ITEMSIZE = {
    TensorProto.FLOAT: 4, TensorProto.UINT8: 1, TensorProto.INT8: 1,
    TensorProto.UINT16: 2, TensorProto.INT16: 2, TensorProto.INT32: 4,
    TensorProto.INT64: 8, TensorProto.STRING: 1, TensorProto.BOOL: 1,
    TensorProto.FLOAT16: 2, TensorProto.DOUBLE: 8, TensorProto.UINT32: 4,
    TensorProto.UINT64: 8, TensorProto.COMPLEX64: 8, TensorProto.COMPLEX128: 16,
    TensorProto.BFLOAT16: 2,
}

MAX_ONNX_BYTES = 1_440_000

def _initializer_elements(tensor: onnx.TensorProto) -> int:
    return int(np.prod(tensor.dims)) if tensor.dims else 1

def _initializer_bytes(tensor: onnx.TensorProto) -> int:
    elements = _initializer_elements(tensor)
    return elements * _DTYPE_ITEMSIZE.get(int(tensor.data_type), 4)

FORBIDDEN_OPS = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function"}

def _forbidden_ops_used(model: onnx.ModelProto) -> list[str]:
    used = []
    for node in model.graph.node:
        if node.op_type in FORBIDDEN_OPS:
            used.append(node.op_type)
    return sorted(set(used))


def _dynamic_shape_issues(model: onnx.ModelProto) -> list[str]:
    issues = []
    def _check_value_info(vi: onnx.ValueInfoProto, label: str):
        tensor_type = vi.type.tensor_type
        for dim_index, dim in enumerate(tensor_type.shape.dim):
            if dim.WhichOneof("value") == "dim_param" or (dim.WhichOneof("value") == "dim_value" and dim.dim_value <= 0):
                issues.append(f"{label} {vi.name!r} dim[{dim_index}] is not a positive integer")
    for vi in model.graph.input:
        _check_value_info(vi, "input")
    for vi in model.graph.output:
        _check_value_info(vi, "output")
    for vi in model.graph.value_info:
        _check_value_info(vi, "intermediate")
    return issues


def _efficiency_summary(model: onnx.ModelProto, model_bytes: int | None = None) -> dict[str, Any]:
    if model_bytes is None:
        model_bytes = len(model.SerializeToString())
    init_count = len(model.graph.initializer)
    init_elements = sum(_initializer_elements(t) for t in model.graph.initializer)
    init_bytes = sum(_initializer_bytes(t) for t in model.graph.initializer)
    histogram: dict[str, int] = {}
    constant_elements = 0
    constant_bytes = 0
    constant_count = 0
    extra_top: list[dict[str, Any]] = []
    for node in model.graph.node:
        histogram[node.op_type] = histogram.get(node.op_type, 0) + 1
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                    tensor = attr.t
                    el = _initializer_elements(tensor)
                    by = _initializer_bytes(tensor)
                    constant_elements += el
                    constant_bytes += by
                    constant_count += 1
                    extra_top.append({
                        "name": node.name or node.output[0] if node.output else "constant",
                        "elements": el,
                        "bytes": by,
                        "shape": list(tensor.dims),
                        "dtype": int(tensor.data_type),
                        "via": "constant_node",
                    })
                    break
    init_top = [{
        "name": t.name,
        "elements": _initializer_elements(t),
        "bytes": _initializer_bytes(t),
        "shape": list(t.dims),
        "dtype": int(t.data_type),
        "via": "initializer",
    } for t in model.graph.initializer]
    top_all = sorted(init_top + extra_top, key=lambda x: x["elements"], reverse=True)[:10]
    parameters = init_elements + constant_elements
    parameter_bytes = init_bytes + constant_bytes
    cost = parameters + model_bytes
    score = max(1.0, 25.0 - math.log(max(cost, 1)))
    forbidden = _forbidden_ops_used(model)
    dyn_issues = _dynamic_shape_issues(model)
    return {
        "bytes": int(model_bytes),
        "parameters": int(parameters),
        "cost": int(cost),
        "score": float(score),
        "validBytes": bool(model_bytes <= MAX_ONNX_BYTES),
        "byteCap": MAX_ONNX_BYTES,
        "graphNodes": len(model.graph.node),
        "initializerCount": init_count + constant_count,
        "initializerBytes": int(parameter_bytes),
        "initializerElements": int(parameters),
        "initializerProtoCount": init_count,
        "constantNodeCount": constant_count,
        "opHistogram": histogram,
        "topInitializers": top_all,
        "forbiddenOps": forbidden,
        "dynamicShapeIssues": dyn_issues,
        "valid": (not forbidden) and (not dyn_issues) and bool(model_bytes <= MAX_ONNX_BYTES),
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


def _layout_positions(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    if not nodes:
        return {}

    node_ids = [n["id"] for n in nodes]
    id_set = set(node_ids)
    original_index = {nid: i for i, nid in enumerate(node_ids)}
    op_of = {n["id"]: (n.get("data") or {}).get("opType", "") for n in nodes}

    producers: dict[str, list[str]] = {nid: [] for nid in node_ids}
    consumers: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src in id_set and tgt in id_set and src != tgt:
            producers[tgt].append(src)
            consumers[src].append(tgt)

    layer: dict[str, int] = {}

    def assign(nid: str, stack: set[str]) -> int:
        if nid in layer:
            return layer[nid]
        if nid in stack:
            return 0
        stack.add(nid)
        if not producers[nid]:
            layer[nid] = 0
        else:
            layer[nid] = max(assign(p, stack) for p in producers[nid]) + 1
        stack.discard(nid)
        return layer[nid]

    for nid in node_ids:
        assign(nid, set())

    for nid in node_ids:
        if op_of[nid] == "Constant" and not producers[nid] and consumers[nid]:
            layer[nid] = max(0, min(layer[c] for c in consumers[nid]) - 1)

    if layer:
        non_output_max = max(
            (layer[nid] for nid in node_ids if op_of[nid] != "Output"),
            default=0,
        )
        for nid in node_ids:
            if op_of[nid] == "Output":
                layer[nid] = max(layer[nid], non_output_max + 1)

    layered: dict[int, list[str]] = {}
    for nid in node_ids:
        layered.setdefault(layer[nid], []).append(nid)

    row: dict[str, int] = {}
    for lay in sorted(layered.keys()):
        ids = layered[lay]
        if lay > 0:
            def key(nid: str) -> tuple[float, int]:
                prods = [p for p in producers[nid] if p in row]
                if prods:
                    mean = sum(row[p] for p in prods) / len(prods)
                else:
                    mean = float(original_index[nid])
                return (mean, original_index[nid])
            ids = sorted(ids, key=key)
            layered[lay] = ids
        for r, nid in enumerate(ids):
            row[nid] = r

    X_SPACING = 220
    Y_SPACING = 130
    X_OFFSET = 70
    Y_OFFSET = 60
    positions: dict[str, dict[str, int]] = {}
    for lay, ids in layered.items():
        for r, nid in enumerate(ids):
            positions[nid] = {"x": X_OFFSET + lay * X_SPACING, "y": Y_OFFSET + r * Y_SPACING}
    return positions


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
    producers: dict[str, tuple[str, int]] = {}
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
        producers[value_info.name] = (node_id, 0)
        index += 1

    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        flat = array.reshape(-1).tolist()
        if array.ndim == 0:
            shape_str = "[]"
        else:
            shape_str = ",".join(str(item) for item in array.shape) or "1"
        node_id = add_node(
            initializer.name,
            f"init_{index}",
            "Constant",
            index,
            {
                "shape": shape_str,
                "sourceName": initializer.name,
                "inputSlots": [],
                "dataType": int(initializer.data_type),
                "values": flat,
                "attrs": {
                    "source": "onnx-initializer",
                    "dataType": int(initializer.data_type),
                    "size": int(array.size),
                    "valuesPreview": flat[:12],
                },
            },
        )
        producers[initializer.name] = (node_id, 0)
        index += 1

    for node_index, onnx_node in enumerate(model.graph.node, start=1):
        slots = [f"in{slot_index}" for slot_index, name in enumerate(onnx_node.input) if name]
        node_data: dict[str, Any] = {
            "sourceName": onnx_node.name,
            "inputSlots": slots,
            "attrs": _onnx_attrs_for_gui(onnx_node),
        }
        if len(onnx_node.output) > 1:
            node_data["outputCount"] = len(onnx_node.output)
        if onnx_node.op_type == "Constant":
            for attr in onnx_node.attribute:
                if attr.name == "value":
                    tensor_proto = helper.get_attribute_value(attr)
                    arr = numpy_helper.to_array(tensor_proto)
                    if arr.ndim == 0:
                        node_data["shape"] = "[]"
                    else:
                        node_data["shape"] = ",".join(str(item) for item in arr.shape) or "1"
                    node_data["dataType"] = int(tensor_proto.data_type)
                    node_data["values"] = arr.reshape(-1).tolist()
                    break
        node_id = add_node(
            onnx_node.name or f"{onnx_node.op_type}_{node_index}",
            f"onnx_{node_index}",
            onnx_node.op_type,
            index,
            node_data,
        )
        index += 1
        visible_slot = 0
        for input_index, input_name in enumerate(onnx_node.input):
            if not input_name:
                continue
            producer_info = producers.get(input_name)
            if producer_info:
                source_id, src_out_idx = producer_info
                slot = f"in{visible_slot}"
                edge = {
                    "id": f"e_{len(edges) + 1}",
                    "source": source_id,
                    "target": node_id,
                    "targetHandle": slot,
                    "data": {"tensor": input_name, "inputIndex": input_index},
                }
                if src_out_idx > 0:
                    edge["sourceHandle"] = f"out{src_out_idx}"
                edges.append(edge)
            visible_slot += 1
        for out_idx, output_name in enumerate(onnx_node.output):
            if output_name:
                producers[output_name] = (node_id, out_idx)

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
        producer_info = producers.get(output.name)
        if producer_info:
            source_id, src_out_idx = producer_info
            edge = {
                "id": f"e_{len(edges) + 1}",
                "source": source_id,
                "target": node_id,
                "targetHandle": "input",
                "data": {"tensor": output.name},
            }
            if src_out_idx > 0:
                edge["sourceHandle"] = f"out{src_out_idx}"
            edges.append(edge)

    positions = _layout_positions(nodes, edges)
    for node in nodes:
        if node["id"] in positions:
            node["position"] = positions[node["id"]]

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


def _arc_grid_to_one_hot_canvas(value: Any, channels: int) -> tuple[np.ndarray, tuple[int, int]]:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D ARC grid for one-hot encoding, got rank {arr.ndim}")
    arr = arr.astype(np.int64)
    height, width = arr.shape
    if height > 30 or width > 30:
        raise ValueError(f"ARC grid shape {[height, width]} exceeds 30x30 canvas")
    canvas = np.zeros([1, channels, 30, 30], dtype=np.float32)
    for c in range(min(channels, 10)):
        canvas[0, c, :height, :width] = (arr == c).astype(np.float32)
    return canvas, (height, width)


def _decode_model_output(actual: np.ndarray) -> np.ndarray:
    arr = np.asarray(actual)
    if arr.ndim == 4 and arr.shape[1] > 1:
        return np.argmax(arr, axis=1, keepdims=True).astype(np.float32)
    return arr


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


def _run_session(
    session: ort.InferenceSession,
    source_grid: Any,
    output_names: list[str] | None = None,
) -> tuple[list[np.ndarray], tuple[int, int]]:
    inputs = session.get_inputs()
    if not inputs:
        raise ValidationError("Phase 1 Strict Equivalence failed: compiled model has no inputs")
    feed = {}
    region: tuple[int, int] | None = None
    for input_meta in inputs:
        shape = list(input_meta.shape)
        if shape == CANVAS_SHAPE:
            canvas, region = _arc_grid_to_canvas(source_grid)
        elif len(shape) == 4 and shape[0] == 1 and shape[2] == 30 and shape[3] == 30:
            canvas, region = _arc_grid_to_one_hot_canvas(source_grid, int(shape[1]))
        else:
            raise ValidationError(
                f"Model input {input_meta.name} has shape {shape}, "
                f"expected {CANVAS_SHAPE} or [1, channels, 30, 30]"
            )
        feed[input_meta.name] = canvas
    assert region is not None
    outputs = session.run(output_names, feed)
    return outputs, region


def _detect_output_region(decoded: np.ndarray, fallback: tuple[int, int]) -> tuple[int, int]:
    arr = np.asarray(decoded)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return fallback
    nonzero = np.argwhere(arr != 0)
    if nonzero.size == 0:
        return fallback
    rmax = int(nonzero[:, 0].max()) + 1
    cmax = int(nonzero[:, 1].max()) + 1
    return (rmax, cmax)


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


_MAX_PREVIEW = 30


def _expose_intermediate_outputs(model: onnx.ModelProto) -> list[str]:
    """Mutate `model` so every node's output tensor surfaces as a graph output.

    Returns the list of newly exposed names (in order).
    """
    existing = {vi.name for vi in model.graph.output}
    added: list[str] = []
    for node in model.graph.node:
        for output_name in node.output:
            if not output_name or output_name in existing:
                continue
            model.graph.output.append(helper.make_empty_tensor_value_info(output_name))
            existing.add(output_name)
            added.append(output_name)
    return added


def _intermediate_preview(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    raw_shape = list(arr.shape)
    dtype_name = str(arr.dtype)

    if arr.dtype == np.bool_:
        numeric = arr.astype(np.int64)
    elif arr.dtype.kind in {"i", "u", "f"}:
        numeric = arr
    else:
        numeric = arr.astype(np.float32, copy=False) if arr.size else np.zeros((0,), dtype=np.float32)

    finite = np.isfinite(numeric) if numeric.dtype.kind == "f" else np.ones_like(numeric, dtype=bool)
    flat = numeric[finite] if finite.any() else numeric
    if flat.size:
        stats = {
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "nnz": int(np.count_nonzero(flat)),
            "size": int(numeric.size),
        }
    else:
        stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "nnz": 0, "size": int(numeric.size)}

    is_approx = False
    grid_arr = numeric

    if grid_arr.ndim == 0:
        grid_arr = grid_arr.reshape(1, 1)
    elif grid_arr.ndim == 1:
        grid_arr = grid_arr.reshape(1, -1)
    elif grid_arr.ndim == 3:
        grid_arr = grid_arr[0]
    elif grid_arr.ndim >= 4:
        if grid_arr.shape[1] > 1:
            grid_arr = np.argmax(grid_arr, axis=1, keepdims=True)
            is_approx = True
        grid_arr = grid_arr[0, 0]

    if grid_arr.ndim > 2:
        grid_arr = grid_arr.reshape(grid_arr.shape[-2], grid_arr.shape[-1])
    if grid_arr.ndim < 2:
        grid_arr = grid_arr.reshape(1, -1)

    h_full, w_full = grid_arr.shape
    truncated = h_full > _MAX_PREVIEW or w_full > _MAX_PREVIEW
    if truncated:
        grid_arr = grid_arr[:_MAX_PREVIEW, :_MAX_PREVIEW]

    rounded = np.rint(grid_arr)
    if np.all(np.isfinite(grid_arr)) and np.all(grid_arr == rounded):
        grid_list: list[list[Any]] = rounded.astype(int).tolist()
    else:
        grid_list = grid_arr.astype(float).tolist()

    return {
        "shape": raw_shape,
        "dtype": dtype_name,
        "stats": stats,
        "grid": grid_list,
        "previewShape": [int(grid_arr.shape[0]), int(grid_arr.shape[1])],
        "truncated": bool(truncated),
        "isApprox": bool(is_approx),
    }


def validate_model(model: onnx.ModelProto, payload: ExportPayload) -> dict[str, str]:
    forbidden = _forbidden_ops_used(model)
    if forbidden:
        raise ValidationError(f"Disallowed ops used: {', '.join(forbidden)}")
    dyn_issues = _dynamic_shape_issues(model)
    if dyn_issues:
        raise ValidationError(f"Dynamic shapes are not allowed: {dyn_issues[0]}")
    if len(model.SerializeToString()) > MAX_ONNX_BYTES:
        raise ValidationError(f"Model exceeds {MAX_ONNX_BYTES} byte cap")

    if not payload.trainingPairs:
        raise ValidationError("Phase 1 Strict Equivalence failed: no ARC training pairs were supplied")

    try:
        session = _make_inference_session(model.SerializeToString())
    except Exception as exc:
        raise ValidationError(f"Phase 1 Strict Equivalence failed: ONNX Runtime could not load model: {exc}") from exc

    strict_outputs: list[np.ndarray] = []
    canvas_outputs: list[np.ndarray] = []

    for index, pair in enumerate(payload.trainingPairs, start=1):
        expected_key = "output" if "output" in pair else "target"
        if "input" not in pair or expected_key not in pair:
            raise ValidationError(f"Phase 1 Strict Equivalence failed: Train {index} is missing input or output grid")
        try:
            raw_outputs, _input_region = _run_session(session, pair["input"])
            actual = _decode_model_output(raw_outputs[0])
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
            raw_outputs, _region = _run_session(session, pair["input"])
            canvas_outputs.append(_decode_model_output(raw_outputs[0]))
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
        model_bytes = len(model.SerializeToString())
        return {
            "status": "compiled",
            "taskId": _task_id(payload),
            "modelBytes": model_bytes,
            "nodeCount": len(payload.nodes),
            "edgeCount": len(payload.edges),
            "io": _model_summary(model),
            "efficiency": _efficiency_summary(model, model_bytes),
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

        trace = bool(payload.traceIntermediates)
        primary_output_name = model.graph.output[0].name if model.graph.output else None
        extra_output_names: list[str] = []
        if trace:
            extra_output_names = _expose_intermediate_outputs(model)

        session = _make_inference_session(model.SerializeToString())
        session_output_names = [meta.name for meta in session.get_outputs()]
        raw_outputs, input_region = _run_session(session, source_grid)
        outputs_by_name = dict(zip(session_output_names, raw_outputs))
        primary_raw = (
            outputs_by_name.get(primary_output_name)
            if primary_output_name is not None
            else raw_outputs[0]
        )
        if primary_raw is None:
            primary_raw = raw_outputs[0]

        actual = _decode_model_output(primary_raw)
        _assert_color_bounds("Run", actual)
        if payload.expectedOutput is not None:
            exp_arr = np.asarray(payload.expectedOutput)
            if exp_arr.ndim == 2:
                out_region = (int(exp_arr.shape[0]), int(exp_arr.shape[1]))
            else:
                out_region = _detect_output_region(actual, input_region)
        else:
            out_region = _detect_output_region(actual, input_region)

        node_outputs: dict[str, Any] = {}
        if trace:
            for name, tensor in outputs_by_name.items():
                if not name.endswith("_out"):
                    continue
                design_id = name[: -len("_out")]
                if not design_id:
                    continue
                node_outputs[design_id] = _intermediate_preview(np.asarray(tensor))

        response = {
            "status": "ran",
            "taskId": _task_id(payload),
            "shape": list(actual.shape),
            "grid": _tensor_to_grid(actual, out_region),
            "io": _model_summary(model),
            "efficiency": _efficiency_summary(model),
        }
        if trace:
            response["nodeOutputs"] = node_outputs
        return response
    except ValidationError as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "failed", "reason": str(exc)})


class BatchExportPayload(BaseModel):
    tasks: list[ExportPayload]


@app.get("/api/baselines-summary")
def baselines_summary():
    out: dict[str, Any] = {}
    for path in sorted(IMPORT_DIR.glob("task*.onnx")):
        if not _IMPORT_NAME_RE.match(path.name):
            continue
        task_id = path.stem
        try:
            model = onnx.load(str(path))
            eff = _efficiency_summary(model)
            out[task_id] = eff
        except Exception as exc:
            out[task_id] = {"error": str(exc)}
    total_score = sum((v.get("score") or 0) for v in out.values() if isinstance(v, dict) and "error" not in v)
    return {"items": out, "count": len(out), "totalScore": total_score}


@app.post("/api/export-zip")
def export_zip(payload: BatchExportPayload):
    import io
    import zipfile
    buffer = io.BytesIO()
    summary = []
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in payload.tasks:
            task_id = _task_id(entry)
            try:
                model = compile_graph(entry)
                model_bytes = model.SerializeToString()
                eff = _efficiency_summary(model, len(model_bytes))
                zf.writestr(f"{task_id}.onnx", model_bytes)
                summary.append({
                    "taskId": task_id,
                    "status": "ok",
                    "bytes": eff["bytes"],
                    "parameters": eff["parameters"],
                    "cost": eff["cost"],
                    "score": eff["score"],
                    "valid": eff["valid"],
                })
            except Exception as exc:
                summary.append({"taskId": task_id, "status": "failed", "reason": str(exc)})
        zf.writestr("summary.json", json.dumps(summary, indent=2))
    buffer.seek(0)
    headers = {
        "Content-Disposition": "attachment; filename=neurogolf-submission.zip",
        "X-Submission-Total": str(len(summary)),
    }
    from starlette.responses import StreamingResponse
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


@app.post("/api/check")
def check_correctness(payload: RunPayload):
    try:
        model = compile_graph(payload)
        eff = _efficiency_summary(model)
        session = _make_inference_session(model.SerializeToString())
        results = []
        all_ok = True
        for index, pair in enumerate(payload.trainingPairs or []):
            try:
                inp = pair.get("input")
                expected = pair.get("output")
                if inp is None or expected is None:
                    results.append({"index": index, "kind": "train", "ok": False, "reason": "missing input/output"})
                    all_ok = False
                    continue
                raw_outputs, _ = _run_session(session, inp)
                actual = _decode_model_output(raw_outputs[0])
                exp_arr = np.asarray(expected)
                h, w = (int(exp_arr.shape[0]), int(exp_arr.shape[1])) if exp_arr.ndim == 2 else (30, 30)
                actual_window = actual[..., :h, :w]
                expected_canvas, _ = _arc_grid_to_canvas(expected)
                expected_window = expected_canvas[..., :h, :w]
                ok = bool(np.array_equal(actual_window, expected_window))
                results.append({"index": index, "kind": "train", "ok": ok, "shape": [int(h), int(w)]})
                if not ok:
                    all_ok = False
            except Exception as exc:
                results.append({"index": index, "kind": "train", "ok": False, "reason": str(exc)})
                all_ok = False
        return {
            "status": "ok",
            "taskId": _task_id(payload),
            "correct": bool(all_ok and len(results) > 0),
            "pairs": results,
            "efficiency": eff,
        }
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

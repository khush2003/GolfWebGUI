import unittest
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import ExportPayload, ValidationError, app, compile_graph, validate_model


client = TestClient(app)


def payload_dict(payload):
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


def identity_payload(output_grid=None):
    source = [[0, 1], [2, 3]]
    return ExportPayload(
        projectName="test",
        taskId="task001",
        nodes=[
            {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,1,30,30"}},
            {"id": "identity_1", "type": "op", "data": {"opType": "Identity", "shape": "1,1,30,30"}},
            {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,1,30,30"}},
        ],
        edges=[
            {"source": "input_1", "target": "identity_1"},
            {"source": "identity_1", "target": "output_1"},
        ],
        trainingPairs=[{"input": source, "output": output_grid or source}],
    )


def color_remap_payload():
    source = [[6, 6, 7, 6], [6, 6, 7, 7], [7, 7, 6, 7]]
    target = [[2, 2, 7, 2], [2, 2, 7, 7], [7, 7, 2, 7]]
    return ExportPayload(
        projectName="color-remap",
        taskId="task276",
        nodes=[
            {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,1,30,30"}},
            {"id": "const_6", "type": "op", "data": {"opType": "Constant", "shape": "1,1,30,30", "value": "6"}},
            {"id": "const_2", "type": "op", "data": {"opType": "Constant", "shape": "1,1,30,30", "value": "2"}},
            {"id": "equal_1", "type": "op", "data": {"opType": "Equal", "shape": "1,1,30,30"}},
            {"id": "where_1", "type": "op", "data": {"opType": "Where", "shape": "1,1,30,30"}},
            {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,1,30,30"}},
        ],
        edges=[
            {"source": "input_1", "target": "equal_1", "targetHandle": "a"},
            {"source": "const_6", "target": "equal_1", "targetHandle": "b"},
            {"source": "equal_1", "target": "where_1", "targetHandle": "condition"},
            {"source": "const_2", "target": "where_1", "targetHandle": "true"},
            {"source": "input_1", "target": "where_1", "targetHandle": "false"},
            {"source": "where_1", "target": "output_1", "targetHandle": "input"},
        ],
        trainingPairs=[{"input": source, "output": target}],
    )


def single_op_payload(op_type, attrs=None, output_grid=None):
    source = [[1, 2, 3], [4, 5, 6]]
    nodes = [
        {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,1,30,30"}},
        {"id": "op_1", "type": "op", "data": {"opType": op_type, "shape": "1,1,30,30", "attrs": attrs or {}}},
        {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,1,30,30"}},
    ]
    edges = [
        {"source": "input_1", "target": "op_1", "targetHandle": "input"},
        {"source": "op_1", "target": "output_1", "targetHandle": "input"},
    ]
    return ExportPayload(
        projectName=f"test-{op_type.lower()}",
        taskId="task001",
        nodes=nodes,
        edges=edges,
        trainingPairs=[{"input": source, "output": output_grid or source}],
    )


def binary_op_payload(op_type, attrs=None):
    return ExportPayload(
        projectName=f"test-{op_type.lower()}",
        taskId="task001",
        nodes=[
            {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,1,30,30"}},
            {"id": "const_1", "type": "op", "data": {"opType": "Constant", "shape": "1,1,30,30", "value": "1"}},
            {"id": "op_1", "type": "op", "data": {"opType": op_type, "shape": "1,1,30,30", "attrs": attrs or {}}},
            {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,1,30,30"}},
        ],
        edges=[
            {"source": "input_1", "target": "op_1", "targetHandle": "a"},
            {"source": "const_1", "target": "op_1", "targetHandle": "b"},
            {"source": "op_1", "target": "output_1", "targetHandle": "input"},
        ],
        trainingPairs=[{"input": [[1]], "output": [[1]]}],
    )


class ServerCompilerTests(unittest.TestCase):
    def test_compile_and_validate_identity_graph(self):
        payload = identity_payload()
        model = compile_graph(payload)
        result = validate_model(model, payload)
        self.assertEqual(result["train"], "passed")
        self.assertEqual(result["shape"], "passed")
        self.assertEqual(result["colors"], "passed")

    def test_rejects_banned_ops(self):
        payload = ExportPayload(
            projectName="bad",
            taskId="task001",
            nodes=[
                {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,1,30,30"}},
                {"id": "loop_1", "type": "op", "data": {"opType": "Loop", "shape": "1,1,30,30"}},
                {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,1,30,30"}},
            ],
            edges=[
                {"source": "input_1", "target": "loop_1"},
                {"source": "loop_1", "target": "output_1"},
            ],
            trainingPairs=[{"input": [[0]], "output": [[0]]}],
        )
        with self.assertRaisesRegex(ValueError, "Banned ONNX operation"):
            compile_graph(payload)

    def test_rejects_malformed_graph_edges(self):
        payload = identity_payload()
        payload.edges = [{"source": "missing", "target": "identity_1"}]
        with self.assertRaisesRegex(ValueError, "edge source"):
            compile_graph(payload)

    def test_validation_failure_reports_mismatch(self):
        payload = identity_payload(output_grid=[[9, 9], [9, 9]])
        model = compile_graph(payload)
        with self.assertRaisesRegex(ValidationError, "Strict Equivalence failed"):
            validate_model(model, payload)

    def test_named_input_slots_compile_color_remap(self):
        payload = color_remap_payload()
        model = compile_graph(payload)
        result = validate_model(model, payload)
        self.assertEqual(result["train"], "passed")

    def test_rejects_duplicate_named_input_slot(self):
        payload = color_remap_payload()
        payload.edges[1]["targetHandle"] = "a"
        with self.assertRaisesRegex(ValueError, "multiple edges for input slot 'a'"):
            compile_graph(payload)

    def test_slice_can_crop_output_window(self):
        payload = single_op_payload(
            "Slice",
            attrs={"starts": [0, 0, 0, 0], "ends": [1, 1, 1, 2], "axes": [0, 1, 2, 3], "steps": [1, 1, 1, 1]},
            output_grid=[[1, 2]],
        )
        model = compile_graph(payload)
        result = validate_model(model, payload)
        self.assertEqual(result["train"], "passed")

    def test_transpose_can_swap_grid_axes(self):
        payload = single_op_payload("Transpose", attrs={"perm": [0, 1, 3, 2]}, output_grid=[[1, 4], [2, 5], [3, 6]])
        model = compile_graph(payload)
        result = validate_model(model, payload)
        self.assertEqual(result["train"], "passed")

    def test_compile_spatial_and_coordinate_nodes(self):
        cases = [
            single_op_payload("Pad", attrs={"pads": [0, 0, 1, 1, 0, 0, 0, 0], "value": 0}),
            single_op_payload("Tile", attrs={"repeats": [1, 1, 1, 1]}),
            single_op_payload("Resize", attrs={"sizes": [1, 1, 30, 30], "mode": "nearest"}),
            single_op_payload("Conv", attrs={"weight_shape": [1, 1, 3, 3], "weights": [1, 0, 0, 0, 0, 0, 0, 0, 0], "pads": [1, 1, 1, 1]}),
            ExportPayload(
                projectName="coords",
                taskId="task001",
                nodes=[
                    {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,1,30,30"}},
                    {"id": "row_1", "type": "op", "data": {"opType": "RowIndex", "shape": "1,1,30,30"}},
                    {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,1,30,30"}},
                ],
                edges=[{"source": "row_1", "target": "output_1", "targetHandle": "input"}],
                trainingPairs=[{"input": [[0]], "output": [[0]]}],
            ),
        ]
        for payload in cases:
            with self.subTest(op=payload.nodes[1]["data"]["opType"]):
                compile_graph(payload)

    def test_concat_compiles_two_inputs(self):
        payload = ExportPayload(
            projectName="concat",
            taskId="task001",
            nodes=[
                {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,1,30,30"}},
                {"id": "const_1", "type": "op", "data": {"opType": "Constant", "shape": "1,1,30,30", "value": "0"}},
                {"id": "concat_1", "type": "op", "data": {"opType": "Concat", "shape": "1,2,30,30", "attrs": {"axis": 1}}},
                {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,2,30,30"}},
            ],
            edges=[
                {"source": "input_1", "target": "concat_1", "targetHandle": "a"},
                {"source": "const_1", "target": "concat_1", "targetHandle": "b"},
                {"source": "concat_1", "target": "output_1", "targetHandle": "input"},
            ],
            trainingPairs=[{"input": [[1]], "output": [[1]]}],
        )
        compile_graph(payload)

    def test_compile_low_risk_extended_ops(self):
        cases = [
            single_op_payload("Relu"),
            single_op_payload("Abs"),
            single_op_payload("Neg"),
            single_op_payload("Floor"),
            single_op_payload("Clip", attrs={"min": 0, "max": 9}),
            single_op_payload("Sign"),
            single_op_payload("Sqrt"),
            single_op_payload("ReduceMax", attrs={"axes": [1], "keepdims": 1}),
            single_op_payload("ReduceMin", attrs={"axes": [1], "keepdims": 1}),
            binary_op_payload("GreaterOrEqual"),
            binary_op_payload("LessOrEqual"),
            binary_op_payload("Add"),
            binary_op_payload("Sub"),
            binary_op_payload("Mul"),
            binary_op_payload("Div"),
            binary_op_payload("Mod"),
            binary_op_payload("Min"),
            binary_op_payload("Max"),
            binary_op_payload("Sum"),
        ]
        for payload in cases:
            with self.subTest(op=payload.nodes[-2]["data"]["opType"]):
                compile_graph(payload)

    def test_generic_in_numbered_slots_order_inputs(self):
        payload = binary_op_payload("Add")
        payload.edges[0]["targetHandle"] = "in0"
        payload.edges[1]["targetHandle"] = "in1"
        compile_graph(payload)

    def test_compile_endpoint_reports_model_summary_without_upload(self):
        response = client.post("/api/compile", json=payload_dict(identity_payload()))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "compiled")
        self.assertGreater(data["modelBytes"], 0)
        self.assertEqual(data["io"]["inputs"][0]["shape"], [1, 1, 30, 30])
        self.assertEqual(data["io"]["outputs"][0]["shape"], [1, 1, 30, 30])

    def test_run_endpoint_runs_current_input_grid_without_upload(self):
        payload = payload_dict(identity_payload())
        payload["inputGrid"] = [[4, 5], [6, 7]]
        response = client.post("/api/run", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ran")
        self.assertEqual(data["grid"], [[4, 5], [6, 7]])

    def test_best_graph_endpoint_imports_onnx_as_visual_nodes(self):
        best_path = Path(__file__).resolve().parents[1] / "client" / "public" / "best" / "onnx" / "task001.onnx"
        if not best_path.exists():
            self.skipTest("best ONNX assets are not present")
        response = client.get("/api/best-graph/task001")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["taskId"], "task001")
        self.assertTrue(data["meta"]["rawOnnx"])
        self.assertGreater(len(data["nodes"]), 0)
        self.assertGreater(len(data["edges"]), 0)
        self.assertIn("Slice", data["meta"]["opTypes"])


if __name__ == "__main__":
    unittest.main()

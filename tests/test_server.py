import unittest

from server import ExportPayload, ValidationError, compile_graph, validate_model


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


if __name__ == "__main__":
    unittest.main()

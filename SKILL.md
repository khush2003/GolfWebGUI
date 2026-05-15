---
name: neurogolf-lab
description: Use NeuroGolf Lab to solve ARC/NeuroGolf tasks through the visual graph workflow or headless agent export. Covers task inspection, graph JSON, backend validation, Hugging Face artifact export, and Kaggle CLI submission discipline.
---

# NeuroGolf Lab

Use this skill when working in the NeuroGolf Lab repo to build, validate, export, or submit ARC/NeuroGolf ONNX candidates.

## Rules

- Use the GUI workflow or `scripts/agent_export.py`; do not hand-write ONNX except for diagnostics.
- `/api/export` is the gate: compile -> ONNX Runtime validation -> Hugging Face upload.
- A candidate is not successful unless backend validation passes.
- Do not print or commit `.env`, tokens, Kaggle credentials, logs, archives, `client/dist/`, or `node_modules/`.
- Do not submit to Kaggle unless the user explicitly asks.

## Human + Agent Workflow

1. Human opens the web GUI and selects a task.
2. Agent inspects task JSON and proposes the smallest graph recipe.
3. Human can build the graph visually, or agent can test the graph headlessly.
4. Agent reports exact nodes, edges, constants, attributes, validation result, and artifact path.
5. Human decides whether to package/submit artifacts.

## Headless Export

Run from repo root while the backend is running:

```bash
python3 scripts/agent_export.py --task task276 --color-remap '{"6":2}'
python3 scripts/agent_export.py --task task010 --graph graph.json
```

`graph.json` may contain either a full export payload or `nodes`/`edges`; the helper fills training pairs from task JSON when absent.

## Useful Node Families

Pixel logic:

```text
Constant Equal Greater Less Not And Where Add Sub Mul Div Cast
```

Global/axis logic:

```text
ReduceSum ArgMax RowIndex ColIndex
```

Spatial logic:

```text
Slice Pad Concat Transpose Tile Resize Conv
```

Default canvas is `[1,1,30,30]`. Prefer simple static graphs and visible-example validation before adding complexity.

## Kaggle Discipline

Use Kaggle CLI only after artifacts are packaged for the competition:

```bash
kaggle competitions submit -c neurogolf-2026 -f submission.zip -m "message"
kaggle competitions submissions -c neurogolf-2026
```

Keep `~/.kaggle/kaggle.json` outside the repo. Record which task artifacts changed and whether public score improved.

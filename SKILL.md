---
name: neurogolf-lab
description: Use NeuroGolf Lab to solve ARC/NeuroGolf tasks through visual graph workflow, headless candidate testing, paper/math research, pipeline scoring, backend validation, Hugging Face artifact export, and Kaggle CLI submission discipline.
---

# NeuroGolf Lab

Use this skill when working in the NeuroGolf Lab repo to research, build, validate, score, export, or submit ARC/NeuroGolf ONNX candidates.

## Rules

- Use the GUI workflow or `scripts/agent_export.py`; do not hand-write ONNX except for diagnostics.
- `/api/export` is the gate: compile -> ONNX Runtime validation -> Hugging Face upload.
- A candidate is not successful unless backend validation passes.
- Do not print or commit `.env`, tokens, Kaggle credentials, logs, archives, private best-attempt ONNX files, `client/dist/`, or `node_modules/`.
- Do not submit to Kaggle unless the user explicitly asks.
- When using online research, cite paper/source URLs in the working notes or final summary, and convert each idea into a concrete graph/test hypothesis.

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

## Research And Pipeline Mode

For paper-driven solving or systematic score improvement, read `references/research_pipeline.md`.

Default loop:

1. Inspect one task or task family.
2. Compare visible examples against current best behavior.
3. Search for one relevant math/program-synthesis idea only when needed.
4. Translate that idea into GUI nodes or a compiler TODO.
5. Score on `train`, `test`, and `arc-gen`; promote only if it improves against the current best.

Do not chase papers abstractly. Every research note must end with:

```text
task family:
paper/source:
math idea:
candidate nodes:
tests to run:
promotion rule:
```

## Useful Node Families

Pixel logic:

```text
Constant Equal Greater Less GreaterOrEqual LessOrEqual Not And Or Xor Where Add Sub Mul Div Mod Min Max Sum Cast
```

Global/axis logic:

```text
ReduceSum ReduceMax ReduceMin ArgMax RowIndex ColIndex
```

Spatial logic:

```text
Slice Pad Concat Transpose Tile Resize Conv
```

Imported best-submission graphs may also contain unsupported ONNX ops. Use visual import to inspect them, then update compiler support deliberately from `COMPILER_TODOS.md`.

If a solution needs a missing node/op, add it deliberately:

1. Add it to the GUI palette and input-slot metadata.
2. Add compiler support or a precise `COMPILER_TODOS.md` entry.
3. Add backend tests and e2e coverage when the node must be selectable.
4. Record partial support in `errors.md` during long runs.

Default canvas is `[1,1,30,30]`. Prefer simple static graphs and visible-example validation before adding complexity.

## Kaggle Discipline

Use Kaggle CLI only after artifacts are packaged for the competition:

```bash
kaggle competitions submit -c neurogolf-2026 -f submission.zip -m "message"
kaggle competitions submissions -c neurogolf-2026
```

Keep `~/.kaggle/kaggle.json` outside the repo. Record which task artifacts changed and whether public score improved.

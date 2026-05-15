# NeuroGolf Research Pipeline

Use this reference when the task needs online paper search, math ideas, pipeline design, or systematic score improvement.

## Research Search Discipline

Search online only for ideas that can become a concrete graph, compiler primitive, or scoring method.

Prefer primary sources:

- arXiv papers
- conference papers
- official ARC/ARC-AGI resources
- official ONNX/ONNX Runtime docs for operator behavior
- original code repos linked from papers

Avoid using blog summaries as the main source unless they point to primary papers.

Good query families:

```text
ARC AGI program synthesis DSL abstraction paper
ARC challenge object centric reasoning connected components
ARC grid transformation program induction
neuro symbolic program synthesis visual reasoning ARC
cellular automata grid transformation induction
morphological image processing connected components holes lines
array program synthesis examples input output grids
DSL search wake sleep ARC tasks
minimum description length program synthesis grid tasks
ONNX operator static graph gather scatter one hot shape inference
```

For every source, record:

```text
source URL:
problem type:
usable idea:
required primitive:
candidate node graph:
why it might generalize:
why it might fail:
```

## Math Ideas To Look For

Map papers into these implementable buckets:

- Object extraction: connected components, bounding boxes, masks, per-object features.
- Symmetry: reflect, rotate, tile, repeat, periodicity, grid axes.
- Morphology: dilation, erosion, hole fill, line extension, neighborhood counts.
- Coordinate logic: row/col masks, between points, distance fields, parity, modular patterns.
- Color algebra: remap, majority/minority color, foreground/background split, palette transfer.
- Compression/MDL: shortest rule that explains examples, not highest train-only fit.
- Search: enumerate small graphs, prune by shape/color constraints, rank by simplicity.

If the idea needs an unsupported primitive, update `COMPILER_TODOS.md` before implementing.

If the primitive is central to the task family, add the node/op instead of weakening the candidate:

```text
1. add GUI palette entry and input slots
2. add compile_graph support or exact TODO if implementation is too large
3. add focused backend tests
4. add e2e coverage if the GUI must expose it
5. log partial support or blockers in errors.md
```

## Headless Testing Pipeline

Treat a candidate as a test artifact:

```text
candidate graph JSON
task JSON
current best ONNX
scoring report
promotion decision
```

Minimum checks:

```bash
curl -fsS http://127.0.0.1:8081/
curl -fsS http://127.0.0.1:8081/tasks/task010.json
curl -fsS http://127.0.0.1:8081/api/best-graph/task010
```

Kaggle auth setup:

```bash
export KAGGLE_USERNAME="..."
export KAGGLE_KEY="..."
/usr/bin/python3.12 scripts/kaggle_auth.py --install --write-from-env
```

Compile candidate:

```bash
curl -fsS http://127.0.0.1:8081/api/compile \
  -H 'Content-Type: application/json' \
  --data @candidate.json
```

Run candidate:

```bash
curl -fsS http://127.0.0.1:8081/api/run \
  -H 'Content-Type: application/json' \
  --data @candidate-run.json
```

Export only after scoring:

```bash
python3 scripts/agent_export.py --task task010 --graph candidate.json
```

Build a Kaggle zip only from promoted reports:

```bash
/usr/bin/python3.12 scripts/build_submission_zip.py \
  --candidate task010=candidate.json \
  --report runs/reports/task010-report.json \
  --message "task010 measured improvement"
```

Submit only through the rate-limit guard:

```bash
/usr/bin/python3.12 scripts/kaggle_guarded_submit.py \
  --zip runs/submissions/submission-123.zip \
  --report runs/reports/task010-report.json \
  --message "measured task010 improvement"
```

The guarded submit wrapper enforces:

- at least 180 seconds between submissions;
- at least one promoted report per zip;
- no direct Kaggle submission from ad hoc shell commands.

## Scoring Rules

Always separate these outcomes:

```text
compile failed
runtime failed
shape failed
exact failed
color bounds failed
exact passed
```

Score at three levels:

```text
exact examples passed / total
cell accuracy over comparable windows
shape agreement
```

Use splits this way:

- `train`: hard gate. A candidate that fails train is not promoted.
- `test`: public-like check. Use when outputs are available.
- `arc-gen`: overfit detector. Prefer candidates that survive here.

Promotion rule:

```text
promote if train passes
and candidate test exact >= current best test exact
and candidate arc-gen exact > current best arc-gen exact
and no new runtime/shape/color failures appear
```

If exact scores tie, prefer the smaller graph and fewer unsupported ops.

## Candidate Notes Template

Use this exact compact format in working notes:

```text
task:
family:
source:
hypothesis:
graph:
train:
test:
arc-gen:
best comparison:
decision:
next:
```

## Agent Workflow

1. Pick one task or one family.
2. Inspect examples and current best graph import.
3. Search papers only if local reasoning stalls or a family-level primitive is missing.
4. Build a candidate graph in GUI JSON shape.
5. Compile and run headlessly.
6. Compare to current best.
7. Export only after promotion rule passes.
8. Record what failed if not promoted.

Do not submit to Kaggle unless explicitly asked.

## Two-Hour Submission Run

Use `configs/headless_hf_kaggle_run.json` as the controller prompt/config for the run.

Use this cadence when asked for a limited run:

```text
00:00-00:10 auth check, server smoke, baseline score report
00:10-00:25 select target task family and first candidate
00:25-01:40 iterate candidates; package only promoted reports
01:40-01:55 guarded submits, max one every 180 seconds
01:55-02:00 summarize score movement and errors.md follow-ups
```

Never try to force exactly 10 submissions if the promotion gate does not produce 10 improved zips.
The target is up to 10 measured submissions, not 10 guesses.

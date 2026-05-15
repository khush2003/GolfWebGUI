# NeuroGolf Lab Agent Notes

Use this repo as an ARC/NeuroGolf solving workbench, not as a place to hand-write ONNX files directly.

## Primary Workflow

1. Inspect task JSON in `client/dist/tasks/` or `client/public/tasks/`.
2. Form a transformation hypothesis.
3. Build a graph using the same node JSON shape the GUI emits.
4. Submit through `/api/export`.
5. Treat backend validation as the gate. A graph is not successful unless validation passes.
6. Hugging Face upload happens only after validation passes.

## Research + Pipeline Workflow

Use `SKILL.md` as the repo skill. For math/paper-driven solving, read `references/research_pipeline.md`.

Agent loop:

1. Pick one task or a small task family.
2. Inspect task JSON and current best ONNX behavior.
3. Search papers or known methods only when they can suggest an implementable graph primitive.
4. Convert the idea into GUI node JSON, not handwritten ONNX.
5. Compile, run, and score headlessly before export.
6. Promote only if the candidate beats the current best on a held-out split or `arc-gen` proxy.

Testing discipline:

- `train` is a hard gate, not a score target.
- `test` and `arc-gen` are used to detect overfit.
- Record exact match, cell accuracy, shape failures, color failures, and runtime failures separately.
- Keep a rejected-candidate note when a research idea fails, so future agents do not retry it blindly.
- Use `scripts/kaggle_guarded_submit.py`; never call `kaggle competitions submit` directly during run loops.
- Keep at least 180 seconds between Kaggle submissions.
- Respect Kaggle's current daily submission limit; the live UI/server counter is the operational source of truth.
- Each submitted zip must be backed by a promoted score report from `scripts/score_candidate.py`.

Node/primitive discipline:

- If a task solution needs a node/op that is missing from the GUI or compiler, add it deliberately.
- Update GUI palette/input slots, compiler support, backend tests, and e2e coverage as needed.
- If full compiler support is too large for the current run, add the limitation to `COMPILER_TODOS.md` and `errors.md`.
- Do not silently replace a missing core primitive with a weaker graph just to keep moving.

## Human + Agent Collaboration

- The human uses the browser GUI to inspect grids, connect nodes, edit constants/attributes, and export.
- The agent may use `scripts/agent_export.py` for headless attempts.
- If the agent finds a useful graph, explain the node/edge recipe so the human can recreate or inspect it in the GUI.
- Do not bypass the compiler by writing ONNX directly except for diagnostic comparison.

## Headless Export

Color remap shortcut:

```bash
python3 scripts/agent_export.py --task task276 --color-remap '{"6":2}'
```

Graph JSON:

```bash
python3 scripts/agent_export.py --task task010 --graph graph.json
```

The helper posts to the same `/api/export` endpoint as the GUI.

## Headless Score And Submit

Use the tuned run config:

```bash
configs/headless_hf_kaggle_run.json
```

Authenticate Kaggle from `.env` or environment variables:

```bash
/usr/bin/python3.12 scripts/kaggle_auth.py --install --write-from-env
```

The current Kaggle CLI also accepts an access-token file at `~/.kaggle/access_token`. Or prompt once without putting the legacy API secret in shell history:

```bash
/usr/bin/python3.12 scripts/kaggle_auth.py --install --interactive
```

Check Hugging Face token/repo access from `.env` or environment variables:

```bash
/usr/bin/python3.12 scripts/hf_auth.py
```

Score a candidate:

```bash
/usr/bin/python3.12 scripts/score_candidate.py --task task010 --graph graph.json
```

Build a submission zip by patching promoted tasks into the private local best zip:

```bash
/usr/bin/python3.12 scripts/build_submission_zip.py \
  --candidate task010=graph.json \
  --report runs/reports/task010-report.json \
  --message "task010 improvement"
```

Submit with promotion and rate-limit guardrails:

```bash
/usr/bin/python3.12 scripts/kaggle_guarded_submit.py \
  --zip runs/submissions/submission.zip \
  --report runs/reports/task010-report.json \
  --message "measured task010 improvement"
```

Do not bypass the guarded submit wrapper. Log blockers in `errors.md`.

## Safety

- Do not print `.env` values.
- Do not commit secrets, tokens, logs, build output, archives, or dependency folders.
- Do not commit private best-attempt artifacts: `client/public/best/manifest.json`, `client/public/best/submission-best.zip`, or `client/public/best/onnx/*.onnx`.
- Keep runtime hosts and tunnels in local deployment notes, not public source docs.
- Run `python3 -m pytest -q` and `cd client && npm run build` before commits.

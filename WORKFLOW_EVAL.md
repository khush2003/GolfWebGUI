# NeuroGolf Lab Workflow Evaluation

Date: 2026-05-13
Active project during evaluation: remote deployment checkout.
Local tunnel convention: use your private SSH or tunnel configuration; do not commit hostnames, IPs, or credentials.

## Verdict

PARTIALLY READY

The workflow can now solve and export a narrow class of same-shape pixelwise ARC transformations through the visual graph model: color remaps built from `Input`, `Constant`, `Equal`, `Where`, and `Output`.

It is not ready for broad real ARC solving. Most inspected tasks need spatial movement, crop/expand, object extraction, connected components, tiling, symmetry, or coordinate-aware masks that are not expressible with the current node set.

## Step 1: Baseline Health Check

- Backend tests before changes: `python3 -m pytest -q` passed, 4 tests.
- Frontend build before changes: `cd client && npm run build` passed.
- Live app before changes: `GET http://127.0.0.1:8081/` returned 200.
- Static task URLs before changes:
  - `/tasks/task001.json`: 200
  - `/tasks/task010.json`: 200
  - `/tasks/task050.json`: 200
  - `/tasks/task100.json`: 200
  - `/tasks/task200.json`: 200
  - `/tasks/task300.json`: 200
  - `/tasks/task400.json`: 200
- Task inventory: 400 JSON files, `task001.json` through `task400.json`.
- Invalid export check: `/api/export` rejected a graph containing banned op `Loop` with HTTP 400 and reason `Banned ONNX operation(s): Loop`.
- Push gate behavior before fixes: invalid graph failed before artifact save/upload. Validated graphs could still fail at upload because the configured HF repo was not accessible.

## Step 2: Real ARC Workflow Test

| Task | Class | Transformation hypothesis | Required primitives | GUI expressible before changes | Compiler exportable before changes | Blocker |
| --- | --- | --- | --- | --- | --- | --- |
| task016 | simple | Fixed color permutation on every cell. | Equality masks, constants, nested `Where`. | No | Partially | GUI had one unnamed input handle and no Constant value control. |
| task276 | simple | Replace color 6 with 2, leave 7 unchanged. | Equality mask, constants, `Where`. | No | Partially | Same as task016. |
| task309 | simple | Replace color 7 with 5, leave 1 and 8 unchanged. | Equality mask, constants, `Where`. | No | Partially | Same as task016. |
| task002 | medium | Fill interior holes in green line/object with yellow. | Neighborhood/object hole detection, local morphology, conditional fill. | No | No | No convolution, shift, padding, or neighborhood aggregate op. |
| task010 | medium | Recolor separate gray line components by position/order. | Component extraction or positional classification, color remap by object. | No | No | No connected components, coordinate masks, or object labels. |
| task050 | medium | Draw green connecting paths between cyan markers. | Marker detection, row/column path fill, between operation. | No | No | No scan/range fill or coordinate comparisons. |
| task001 | shape/crop/position | Expand 3x3 input into a 9x9 tiled/structured pattern. | Resize/tile/repeat or coordinate remap. | No | No | No shape expansion or spatial copy op. |
| task100 | shape/crop/position | Extract/convert object property into 2x2 color block. | Object recognition, crop/summary, output shape control. | No | No | Validator accepts covering output window, but GUI/compiler cannot crop or synthesize shape-dependent block. |
| task300 | probably not solvable | Select one object/pattern from several and crop it. | Object segmentation, selection criteria, crop. | No | No | No object-level ops or dynamic crop. |
| task400 | probably not solvable | Select/reconstruct a 5x5 pattern from a 24x24 scene with competing objects. | Object parsing, comparison, crop, color rule. | No | No | Requires object-level reasoning and spatial extraction. |

## Step 3: Actual Export Attempts

Attempted exports through `/api/export` using graph JSON equivalent to what the frontend emits.

- `task276`: graph `Input -> Equal(input, const 6) -> Where(mask, const 2, input) -> Output`.
  - Backend validation: passed.
  - Final export after HF repo fix: HTTP 200, `task276.onnx` uploaded to `clarkkitchen22/neurogolf-handcrafted`.
- `task309`: graph `Input -> Equal(input, const 7) -> Where(mask, const 5, input) -> Output`.
  - Backend validation: passed.
  - Before HF repo fix: upload failed because repo namespace was wrong.
- `task016`: nested fixed color permutation using eight `Equal`/`Where` stages.
  - Backend validation: passed.
  - Before HF repo fix: upload failed because repo namespace was wrong.

Failure classification:

- Pre-fix GUI blocker: GUI could not reliably build multi-input ops because handles were unnamed.
- Pre-fix GUI blocker: `Constant` node had no explicit value editor.
- Pre-fix backend blocker: multi-input edge ordering depended on target handle strings but did not validate named slots.
- Pre-fix deployment blocker: `HF_REPO_ID` pointed to a namespace the supplied token could not create under.
- Pre-fix error clarity blocker: upload failures looked like failed validation/export instead of reporting validation passed but push failed.

## Step 4: Changes Made

Every change is tied to a blocker above.

1. Added named frontend input handles for supported multi-input ops:
   - binary ops use `a` and `b`;
   - `Where` uses `condition`, `true`, `false`;
   - unary/output ops use `input`.
2. Added backend input slot ordering and duplicate/unknown slot validation.
3. Added frontend `Constant value` inspector control.
4. Replaced the invalid default graph with safe `Input -> Output` using shape `[1,1,30,30]`.
5. Added an export payload preview panel so the GUI user can inspect exactly what will be posted.
6. Added focused compiler tests for named input slots and a real color-remap validation case.
7. Changed `/api/export` to return `upload_failed` with validation details when validation passes but HF push fails.
8. Changed `/api/export` to create the configured private HF model repo with `exist_ok=True` after validation and before upload.
9. Updated local and server `.env` `HF_REPO_ID` to `clarkkitchen22/neurogolf-handcrafted`; token value is intentionally not recorded here.
10. Added `AGENTS.md` with the server path and SSH/tunnel convention.

## Step 6: Verification After Changes

- Backend tests after changes: `python3 -m pytest -q` passed, 6 tests.
- Frontend build after changes: `cd client && npm run build` passed.
- Live app after restart: `GET http://127.0.0.1:8081/` returned 200.
- Export path:
  - invalid `Loop` graph rejected with HTTP 400;
  - `task276` valid graph returned HTTP 200 and uploaded `task276.onnx`;
  - `task309` and `task016` validated locally, and endpoint reached upload phase before repo fix.
- Ignored/untracked safety check: `.env`, `client/dist/`, `client/node_modules/`, logs, caches, and archives are ignored. Nothing was staged.

## Remaining Top 5 Blockers

1. No spatial movement primitives: no crop, pad, slice, tile, resize, transpose, roll/shift, gather/scatter, or coordinate-grid construction.
2. No neighborhood/object primitives: no convolution, connected components, bounding boxes, hole fill, line extension, or object selection.
3. No in-GUI output preview/run panel for graph outputs before export; current validation only happens at backend export.
4. No graph persistence/load implementation despite visible project controls.
5. Shape handling is fixed around `[1,1,30,30]`; real crop/expand ARC tasks need explicit output-window and shape semantics.

## Commands Run

Representative commands:

```bash
python3 -m pytest -q
cd client && npm run build
curl -s -o /tmp/appcheck -w '%{http_code} %{size_download}' http://127.0.0.1:8081/
python3 scripts/agent_export.py --task task276 --color-remap '{"6":2}'
```

Python one-off scripts were used over SSH to inspect task JSONs, post `/api/export` payloads, and validate candidate graphs locally through `compile_graph()` and `validate_model()`.

## Artifact Result

At least one ONNX artifact successfully passed validation and pushed:

- `task276.onnx`
- repo: `clarkkitchen22/neurogolf-handcrafted`
- endpoint response: `{"status":"passed","artifact":"task276.onnx","repo":"clarkkitchen22/neurogolf-handcrafted","path":"task276.onnx","validation":{"train":"passed","shape":"passed","colors":"passed"}}`

## Git Diff Summary

Changed source files:

- `AGENTS.md`
- `WORKFLOW_EVAL.md`
- `client/src/main.jsx`
- `client/src/style.css`
- `server.py`
- `tests/test_server.py`

Local/server `.env` was edited for runtime configuration and remains ignored. No secrets are recorded in this evaluation file.

## Follow-up: Topbar Workflow Button Validation

Date: 2026-05-13

Issue found:

- The visible `Train`, `Test`, `Extra 10`, `Compile`, and `Run` buttons were placeholders with no click handlers, so the actual GUI workflow could not switch examples, compile, or run the current graph from the topbar.

Changes made:

- Added frontend handlers for `Train`, `Test`, and `Extra 10`.
- Added `/api/compile` to compile the current graph without validating against training pairs or pushing artifacts.
- Added `/api/run` to compile the current graph, run it against the currently displayed input grid, and return a preview grid without pushing artifacts.
- Wired `Compile` and `Run` buttons to those backend endpoints.
- Added focused backend tests for `/api/compile` and `/api/run`.

Verification:

- `pytest -q` passed: 12 tests and 5 subtests.
- `cd client && npm run build` passed.
- Live app `GET http://127.0.0.1:8081/` returned 200 after restart.
- Live task JSON `GET http://127.0.0.1:8081/tasks/task010.json` returned 200.
- Browser workflow validation clicked `Train`, `Test`, `Extra 10`, `Compile`, and `Run`; `Compile` reported success and `Run` displayed `RUN OUTPUT`.
- Browser workflow validation also added a `Constant` node through the quick-add UI and deleted it with the keyboard `Delete` key.

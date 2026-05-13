# NeuroGolf Lab

Private WebGL visual node editor and FastAPI backend for building ONNX graph artifacts for NeuroGolf 2026 ARC tasks.

## Run

```bash
python3 -m pip install -r requirements.txt
cd client
npm install
npm run build
cd ..
bash start.sh
```

## Required Environment

Create `.env` in the project root. Do not commit it.

```bash
HF_TOKEN="replace-me"
HF_REPO_ID="ashhhhhh26/neurogolf-handcrafted"
HOST="0.0.0.0"
PORT="8081"
PUBLIC_HOSTNAME="golf.upsidedownatlas.com"
```

Optional local/private variables:

```bash
CLOUDFLARE_API_TOKEN="replace-me"
GITHUB_TOKEN="replace-me"
```

## Notes

- Task JSON files live in `client/public/tasks/` and are lazy-loaded by task id.
- The frontend supports `task001` through `task400`.
- `/api/export` compiles ONNX in memory, validates with ONNX Runtime, and pushes passing artifacts to Hugging Face.
- `.env`, build output, logs, archives, and dependency folders are intentionally ignored.

# Laya Multilingual API

Local FastAPI provider for the [Laya](https://huggingface.co/convaiinnovations/laya)
multilingual decision model (`convaiinnovations/laya`, subfolder `multilingual`,
~645 MB). Runs fully offline once the checkpoint is cached.

## Local run

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt          # add --index-url https://package-mirror.liara.ir/repository/pypi/simple/ if pypi.org is slow
python scripts/download_model.py                       # ~645 MB, resumable; HF_ENDPOINT=https://hf-mirror.com ... to force the mirror
HF_HUB_OFFLINE=1 .venv/Scripts/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker compose up --build -d
```

The build is fully mirror-friendly and needs no HF access:

- base image is pulled through the Liara docker mirror
  (`docker-mirror.liara.ir/library/python:3.12-slim` via the `BASE_IMAGE` arg)
- pip uses the Liara PyPI mirror
- the torch CPU wheel is vendored in `docker-wheels/` (the pytorch.org CDN
  stalls inside the container network on some hosts; refetch on the host if
  you bump torch)
- the checkpoint is staged from `.hf-cache` into `docker-model/` — refresh it
  with `python scripts/stage_model.py` after re-downloading the model

Both staging dirs are gitignored.

## Endpoints

- `GET /health` → `{"status": "ok"}`
- `POST /predict` → `{"state": <string or object>, "questions": {<typed questions>}}`

Example:

```bash
curl -s localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "state": {"body": "We were billed twice for March, refund or we cancel."},
  "questions": {
    "department": {"type": "choice", "instructions": "Which department?",
                   "criteria": {"billing": "invoices, refunds", "technical": "bugs", "other": "rest"}},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to leave?"}
  }
}'
```

## Mirrors & troubleshooting

- PyPI: `PIP_INDEX_URL=https://package-mirror.liara.ir/repository/pypi/simple/` (set in the Dockerfile)
- Hugging Face: default endpoint works here; the Xet CDN host
  (`us.aws.cdn.hf.co`) must resolve. If it stops, retry the download with
  `HF_ENDPOINT=https://hf-mirror.com python scripts/download_model.py`.
- `USE_TF=0` is mandatory (set in `app.py`/Dockerfile): transformers' TF probe
  can deadlock model construction when TensorFlow is installed.
- The server refuses to start without `HF_HUB_OFFLINE=1` so it never silently
  reaches for the network; first `/predict` after start loads the weights
  (~40 s cold, <1 s warm on CPU).

import os

# Must be set before transformers/torch import: transformers probes for
# TensorFlow at import time, and when TF is installed its abseil runtime
# can deadlock model construction (documented laya failure mode).
os.environ.setdefault("USE_TF", "0")

# Keep the model cache inside the project (created by scripts/download_model.py)
# so local runs and Docker agree on where the weights live.
os.environ.setdefault("HF_HOME", str(os.path.join(os.path.dirname(__file__), ".hf-cache")))

from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import laya

MODEL_REPO = "convaiinnovations/laya"
MODEL_SUBFOLDER = "multilingual"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("HF_HUB_OFFLINE"):
        raise RuntimeError(
            "Refusing to start with hub access enabled. Pre-download the "
            "checkpoint first: python scripts/download_model.py, then set "
            "HF_HUB_OFFLINE=1."
        )
    yield


class PredictRequest(BaseModel):
    # Laya accepts a raw state string or a JSON state object.
    state: str | dict
    questions: dict = Field(description="Typed questions; see laya docs for schemas")


@lru_cache(maxsize=1)
def get_agent():
    # First call downloads from HF_HOME (or the mirror via HF_ENDPOINT) if the
    # checkpoint is not cached yet; with HF_HUB_OFFLINE=1 it loads local-only.
    return laya.load(
        MODEL_REPO,
        subfolder=MODEL_SUBFOLDER,
    )


app = FastAPI(title="Laya Multilingual API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        agent = get_agent()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"model not loadable: {exc}") from exc
    try:
        return agent.predict(request.state, request.questions)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"prediction failed: {exc}") from exc

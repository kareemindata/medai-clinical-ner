from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .ner import DEFAULT_MODEL, build_pipeline, extract

app = FastAPI(title="medai-clinical-ner", version="0.1.0")


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    model: str | None = None


@lru_cache(maxsize=4)
def _pipe(model_name: str):
    return build_pipeline(model_name)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
def extract_endpoint(req: ExtractRequest):
    model_name = req.model or DEFAULT_MODEL
    entities = extract(req.text, pipe=_pipe(model_name))
    return {"model": model_name, "entities": [asdict(e) for e in entities]}

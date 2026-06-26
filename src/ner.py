from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Callable, Iterable

DEFAULT_MODEL = os.environ.get("MEDAI_NER_MODEL", "d4data/biomedical-ner-all")


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    score: float


def normalize_spans(raw: Iterable[dict]) -> list[Entity]:
    """Merge subword tokens and drop overlaps, keeping the higher-scoring span."""
    cleaned: list[Entity] = []
    for r in raw:
        label = r.get("entity_group") or r.get("entity") or "MISC"
        text = r.get("word", "").replace(" ##", "").replace("##", "")
        cleaned.append(Entity(
            text=text.strip(),
            label=label,
            start=int(r["start"]),
            end=int(r["end"]),
            score=float(r["score"]),
        ))
    cleaned.sort(key=lambda e: (e.start, -e.score))
    deduped: list[Entity] = []
    for e in cleaned:
        if deduped and e.start < deduped[-1].end:
            if e.score > deduped[-1].score:
                deduped[-1] = e
            continue
        deduped.append(e)
    return deduped


def build_pipeline(model_name: str = DEFAULT_MODEL) -> Callable[[str], list[dict]]:
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("token-classification", model=model, tokenizer=tok, aggregation_strategy="simple")


def extract(text: str, pipe: Callable[[str], list[dict]] | None = None, model_name: str = DEFAULT_MODEL) -> list[Entity]:
    if pipe is None:
        pipe = build_pipeline(model_name)
    return normalize_spans(pipe(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    entities = extract(args.text, model_name=args.model)
    print(json.dumps([asdict(e) for e in entities], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

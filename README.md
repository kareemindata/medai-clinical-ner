# medai-clinical-ner

Lightweight **biomedical / clinical Named Entity Recognition** demo built on
Hugging Face Transformers. Loads a pretrained biomedical NER model, runs it
on free-text clinical notes, and exposes both a CLI and a small FastAPI
service.

> **Disclaimer.** Educational use only. Do not feed real patient data
> (PHI) into hosted model APIs without proper de-identification and
> institutional approval.

## What it does

- Wraps a pretrained NER pipeline (default: `d4data/biomedical-ner-all`).
- Normalizes entity spans, merges sub-tokens, deduplicates overlaps.
- CLI: `python -m src.ner --text "..."` prints JSON entities.
- API: `uvicorn src.app:app --reload` serves `POST /extract`.
- Optional batch script over a folder of `.txt` notes.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# First call downloads the model (~500 MB) to the HF cache.
python -m src.ner --text "Patient is a 64-year-old female with history of type 2 diabetes and hypertension, currently on metformin 1000 mg and lisinopril 20 mg."

uvicorn src.app:app --reload --port 8000
# POST http://localhost:8000/extract  body: {"text": "..."}
```

Switch model via `--model` or the `MEDAI_NER_MODEL` env var.

## Project layout

```
src/
  ner.py        # core extractor + CLI
  app.py        # FastAPI service
  batch.py      # batch over a folder of .txt files
tests/
  test_ner.py   # mocks the HF pipeline, tests span normalization
```

## Suggested models

| model | notes |
| --- | --- |
| `d4data/biomedical-ner-all` | broad biomedical entity coverage |
| `Clinical-AI-Apollo/Medical-NER` | clinical-note tuned |
| `blaze999/Medical-NER` | DeBERTa-based, strong on drugs/diseases |

## Caveats

- All listed models are English-only. Arabic / multilingual clinical NER
  needs different backbones (e.g. `aubmindlab/bert-base-arabertv02` +
  domain fine-tuning).
- Entity schemas differ between models. Don't compare F1 across them
  without aligning label sets.

## License

MIT

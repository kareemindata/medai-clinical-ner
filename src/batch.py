import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .ner import DEFAULT_MODEL, build_pipeline, extract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--glob", default="*.txt")
    args = parser.parse_args()

    pipe = build_pipeline(args.model)
    out = {}
    for path in sorted(Path(args.input_dir).glob(args.glob)):
        text = path.read_text(encoding="utf-8")
        out[path.name] = [asdict(e) for e in extract(text, pipe=pipe)]

    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {len(out)} files -> {args.output}")


if __name__ == "__main__":
    main()

from src.ner import normalize_spans, extract


def test_normalize_merges_overlap_keeps_higher_score():
    raw = [
        {"entity_group": "DRUG", "word": "metformin", "start": 10, "end": 19, "score": 0.95},
        {"entity_group": "MISC", "word": "metform", "start": 10, "end": 17, "score": 0.40},
        {"entity_group": "DOSE", "word": "1000 mg", "start": 20, "end": 27, "score": 0.88},
    ]
    out = normalize_spans(raw)
    assert [e.label for e in out] == ["DRUG", "DOSE"]
    assert out[0].score == 0.95


def test_extract_uses_injected_pipe():
    def fake_pipe(text):
        return [{"entity_group": "DISEASE", "word": "diabetes", "start": 0, "end": 8, "score": 0.99}]
    out = extract("diabetes", pipe=fake_pipe)
    assert len(out) == 1
    assert out[0].label == "DISEASE"
    assert out[0].text == "diabetes"

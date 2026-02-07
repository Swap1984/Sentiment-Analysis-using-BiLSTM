from src.preprocess import clean_text


def test_clean_text_basic():
    s = "Check THIS!!! Visit https://example.com now!!!   "
    out = clean_text(s)
    assert "https" not in out
    assert out.endswith("!") or "!" in out
    assert "  " not in out

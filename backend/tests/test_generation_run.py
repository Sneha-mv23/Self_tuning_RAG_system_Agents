from app.evals.generation_run import GenerationCache, is_refusal, make_key, percentile


def test_key_is_deterministic_and_sensitive():
    a = make_key("m", "u", 0.0, 256, "prompt")
    assert a == make_key("m", "u", 0.0, 256, "prompt")
    assert a != make_key("m", "u", 0.0, 256, "prompt2")
    assert a != make_key("m", "u", 0.7, 256, "prompt")
    assert a != make_key("other", "u", 0.0, 256, "prompt")


def test_cache_roundtrip(tmp_path):
    cache = GenerationCache(str(tmp_path / "c.sqlite"))
    assert cache.get("k") is None
    cache.put("k", "an answer", 1.5)
    assert cache.get("k") == ("an answer", 1.5)
    cache.close()


def test_percentile():
    assert percentile([], 95) == 0.0
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == 5


def test_refusal_detection():
    assert is_refusal("I don't know based on the provided documents.")
    assert is_refusal("I don\u2019t know based on the provided documents.")
    assert not is_refusal("Use the decorator @app.middleware.")
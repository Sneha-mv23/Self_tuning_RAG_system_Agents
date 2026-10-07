from app.evals.generation_run import STALL_S,GenerationRecord,GenerationCache, is_refusal, make_key, percentile, summarize_generation


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


def _rec(id, latency):
    return GenerationRecord(id=id, type="factual", question="q", answer="a", prompt_tokens=10,
                            latency_s=latency, cache_hit=False, over_context=False, error=None, used_chunks=[])


def test_stalled_calls_excluded_from_latency_stats():
    s = summarize_generation([_rec("a", 90.0), _rec("b", 110.0), _rec("c", STALL_S + 1)])
    assert s["mean_latency_s"] == 100.0
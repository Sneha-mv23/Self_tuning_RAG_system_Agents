from urllib.parse import urlparse

from app.config import settings
from app.llm import get_judge

print(f"{settings.judge_model} @ {urlparse(settings.judge_base_url).netloc}")
try:
    reply = get_judge(max_tokens=50).invoke("Reply with exactly one word: OK")
    print("call succeeded:", repr(reply.content))
except Exception as e:  # noqa: BLE001
    print("call failed:\n", str(e)[:1500])
from langchain_openai import ChatOpenAI

from app.config import settings


def get_generator(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.generator_base_url,
        api_key=settings.generator_api_key,
        model=settings.generator_model,
        temperature=temperature,
    )


def get_judge() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.judge_base_url,
        api_key=settings.judge_api_key,
        model=settings.judge_model,
        temperature=0.0,  # judges should be deterministic
    )
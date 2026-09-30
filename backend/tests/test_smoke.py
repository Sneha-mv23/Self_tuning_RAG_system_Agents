from app.config import settings


def test_settings_load():
    assert settings.embedding_model
    assert settings.database_url

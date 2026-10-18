
import pytest
import requests

from documents import rag_pipeline


def test_ollama_missing_model(monkeypatch):
    """Test that a missing Ollama model is handled."""

    monkeypatch.setattr(rag_pipeline, "OLLAMA_MODEL", "")

    monkeypatch.setattr(
        "documents.rag_pipeline.os.getenv",
        lambda key, default="": (
            "" if key == "OLLAMA_MODEL" else default
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="OLLAMA_MODEL is missing",
    ):
        rag_pipeline.generate_answer_ollama(
            "What are the technical skills?",
            [{"page": 1, "text": "Python and SQL"}],
        )


def test_ollama_connection_error(monkeypatch):
    """Test that an Ollama connection failure is handled."""

    monkeypatch.setattr(
        rag_pipeline,
        "OLLAMA_MODEL",
        "llama3.2:3b",
    )

    def fake_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError(
            "Service unavailable"
        )

    monkeypatch.setattr(
        rag_pipeline.requests,
        "post",
        fake_post,
    )

    with pytest.raises(
        RuntimeError,
        match="Cannot connect to Ollama",
    ):
        rag_pipeline.generate_answer_ollama(
            "What are the technical skills?",
            [{"page": 1, "text": "Python and SQL"}],
        )


def test_ollama_empty_context():
    """Test that empty PDF context is handled gracefully."""

    answer = rag_pipeline.generate_answer_ollama(
        "What are the technical skills?",
        [],
    )

    assert "couldn't find relevant information" in answer.lower()


def test_ollama_model_not_found(monkeypatch):
    """Test handling of an unavailable model or endpoint."""

    monkeypatch.setattr(
        rag_pipeline,
        "OLLAMA_MODEL",
        "model-that-does-not-exist",
    )

    class FakeResponse:
        status_code = 404
        text = "Model not found"

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(
                "404 Client Error: Not Found"
            )

        def json(self):
            return {}

    monkeypatch.setattr(
        rag_pipeline.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(
        RuntimeError,
        match="endpoint or model not found",
    ):
        rag_pipeline.generate_answer_ollama(
            "What are the technical skills?",
            [{"page": 1, "text": "Python and SQL"}],
        )


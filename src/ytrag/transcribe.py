"""Transcription backends: any OpenAI-compatible /v1/audio/transcriptions API (Groq, OpenAI)."""
from __future__ import annotations

import io
from typing import Protocol

from .config import Config


class Transcriber(Protocol):
    def transcribe(self, audio: bytes, filename: str, language: str | None = None) -> str: ...


class OpenAICompatTranscriber:
    """Works with Groq and OpenAI (both expose /v1/audio/transcriptions, Whisper models)."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def transcribe(self, audio: bytes, filename: str, language: str | None = None) -> str:
        buf = io.BytesIO(audio)
        buf.name = filename  # SDK infers content type from the file name
        kwargs = {"language": language} if language else {}
        resp = self._client.audio.transcriptions.create(
            model=self.model, file=buf, **kwargs
        )
        return resp.text


def make_transcriber(cfg: Config) -> Transcriber:
    return OpenAICompatTranscriber(
        cfg.transcription_key(),
        cfg.transcription_model,
        cfg.openai_compat_base_url(cfg.transcription_provider),
    )

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass


VOICE_PATIENT_FEMALE = os.environ.get("TTS_VOICE_PATIENT_FEMALE", "bf_isabella")
VOICE_PATIENT_MALE = os.environ.get("TTS_VOICE_PATIENT_MALE", "bm_lewis")
VOICE_DOCTOR = os.environ.get("TTS_VOICE_DOCTOR", "bm_george")


@dataclass(frozen=True)
class TTSResult:
    audio: bytes
    duration_seconds: str | None
    content_type: str


def tts_disabled() -> bool:
    return os.environ.get("DISABLE_TTS", "").lower() in {"1", "true", "yes"}


def tts_url() -> str:
    return os.environ.get("TTS_URL", "https://arjun4707-poirot-kokoro-tts.hf.space").rstrip("/")


def voice_for_speaker(speaker: str, patient_sex: str | None = None) -> str:
    if speaker == "doctor":
        return VOICE_DOCTOR
    if speaker == "patient":
        return VOICE_PATIENT_FEMALE if patient_sex == "female" else VOICE_PATIENT_MALE
    raise ValueError(f"unknown speaker: {speaker}")


def speak_via_tts_service(text: str, voice: str) -> TTSResult:
    payload = (f'{{"text":{_json_string(text)},"voice":{_json_string(voice)}}}').encode("utf-8")
    request = urllib.request.Request(
        f"{tts_url()}/speak",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return TTSResult(
                audio=response.read(),
                duration_seconds=response.headers.get("X-Duration-Seconds"),
                content_type=response.headers.get("Content-Type", "audio/wav"),
            )
    except urllib.error.URLError as exc:
        raise RuntimeError(f"TTS service unavailable: {exc}") from exc


def _json_string(value: str) -> str:
    import json

    return json.dumps(value)

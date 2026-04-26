from __future__ import annotations

from clinical_rl.tts import voice_for_speaker


def test_voice_for_speaker_keeps_patient_and_doctor_voices_separate():
    assert voice_for_speaker("patient", patient_sex="female") == "bf_isabella"
    assert voice_for_speaker("patient", patient_sex="male") == "bm_lewis"
    assert voice_for_speaker("doctor", patient_sex="female") == "bm_george"


def test_voice_for_speaker_rejects_unknown_speaker():
    try:
        voice_for_speaker("narrator", patient_sex="female")
    except ValueError as exc:
        assert "unknown speaker" in str(exc)
    else:
        raise AssertionError("unknown speaker should be rejected")

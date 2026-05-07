# ─────────────────────────────────────────────────────────────────────────────
# MoodSyncAI – models/audio_transcriber.py
#
# PURPOSE:
#   Transcribe a short audio clip to text using OpenAI Whisper.
#   This is the AUDIO branch of our multimodal system.
#
# MODEL USED:
#   openai/whisper-base  (~140MB, downloads automatically on first run)
#   Fast enough for short clips (< 60s), good accuracy for clear speech.
#
# HOW IT WORKS:
#   1. Takes an audio file path (wav/mp3/m4a/ogg)
#   2. Loads it with librosa → converts to 16kHz mono float32
#   3. Whisper processor tokenises the audio
#   4. Whisper model generates the transcript token by token
#   5. Returns the transcribed text string + metadata
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

# ── Module-level cache ────────────────────────────────────────────────────────
_processor = None
_model     = None
_WHISPER_MODEL = "openai/whisper-base"


def _load_model():
    """Load Whisper processor and model — cached after first call."""
    global _processor, _model

    if _processor is None or _model is None:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        print(f"[AudioTranscriber] Loading Whisper model: {_WHISPER_MODEL}")
        print("[AudioTranscriber] First run downloads ~140MB — please wait…")
        _processor = WhisperProcessor.from_pretrained(_WHISPER_MODEL)
        _model     = WhisperForConditionalGeneration.from_pretrained(_WHISPER_MODEL)
        _model.eval()
        print("[AudioTranscriber] ✅ Whisper loaded!")

    return _processor, _model


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe an audio file to text using Whisper.

    Args:
        audio_path : str — path to audio file (wav / mp3 / m4a / ogg)

    Returns a dict:
        {
            "transcript": "No I think the project is going really well.",
            "language":   "en",
            "error":      None
        }
    """

    if audio_path is None:
        return {"transcript": "", "language": "unknown", "error": "No audio file provided"}

    try:
        import librosa

        # ── Step 1: Load audio → 16kHz mono float32 ──────────────────────────
        print(f"[AudioTranscriber] Loading audio: {audio_path}")
        audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Clip to 30s max (Whisper's input limit)
        max_samples = 30 * 16000
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]
            print("[AudioTranscriber] ⚠️ Audio trimmed to 30 seconds")

        # ── Step 2: Load Whisper ──────────────────────────────────────────────
        processor, model = _load_model()

        # ── Step 3: Process audio → input features ────────────────────────────
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features

        # ── Step 4: Generate transcript ───────────────────────────────────────
        import torch
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="en",        # force English
                task="transcribe",
                max_new_tokens=200,
            )

        # ── Step 5: Decode tokens → text ──────────────────────────────────────
        transcript = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0].strip()

        print(f"[AudioTranscriber] ✅ Transcript: {transcript}")

        return {
            "transcript": transcript,
            "language":   "en",
            "error":      None,
        }

    except Exception as e:
        print(f"[AudioTranscriber] ❌ Error: {e}")
        return {
            "transcript": "",
            "language":   "unknown",
            "error":      str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run directly to verify:
#   python models/audio_transcriber.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("AudioTranscriber — no test audio available in standalone mode.")
    print("Test it via app.py by uploading a short audio clip.")
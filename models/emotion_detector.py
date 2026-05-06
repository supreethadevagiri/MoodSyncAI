# ─────────────────────────────────────────────────────────────────────────────
# MoodSyncAI – models/emotion_detector.py
#
# PURPOSE:
#   Detect emotions from a face photo using DeepFace (pretrained CNN).
#   This is the VISUAL branch of our multimodal system.
#
# WHAT IT DETECTS:
#   angry, disgust, fear, happy, sad, surprise, neutral
#
# HOW IT WORKS:
#   1. Takes a PIL Image (what Gradio gives us)
#   2. Converts it to NumPy array (what DeepFace needs)
#   3. DeepFace runs its internal CNN → returns emotion scores
#   4. We normalise scores to probabilities (0.0 – 1.0)
#   5. Returns a clean dict + the top emotion + confidence
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from PIL import Image
from deepface import DeepFace

from config import (
    FACE_DETECTOR_BACKEND,
    EMOTION_LABEL_MAP,
    EMOTION_VALENCE_MAP,
)
from utils.helpers import get_top_label


def analyse_emotion(image: Image.Image) -> dict:
    """
    Analyse facial emotion from a PIL Image.

    Args:
        image : PIL.Image.Image — the uploaded face photo

    Returns a dict with:
        {
            "scores":       {"happy": 0.68, "sad": 0.12, ...},  ← all 7 emotions
            "top_emotion":  "happy",                             ← winning emotion
            "confidence":   0.68,                               ← its score (0-1)
            "display_label": "Happy 😄",                        ← nice label for UI
            "valence":      "positive",                         ← positive/neutral/negative
            "error":        None                                ← error message if failed
        }
    """

    # ── Step 1: Convert PIL Image → NumPy array ───────────────────────────────
    # DeepFace expects a NumPy array in RGB format
    img_array = np.array(image.convert("RGB"))

    # ── Step 2: Run DeepFace emotion analysis ─────────────────────────────────
    try:
        result = DeepFace.analyze(
            img_path       = img_array,
            actions        = ["emotion"],          # we only want emotion, not age/gender
            detector_backend = FACE_DETECTOR_BACKEND,
            enforce_detection = False,             # don't crash if face not perfectly detected
            silent         = True,                 # suppress DeepFace console logs
        )

        # DeepFace returns a list — take the first (most prominent) face
        face_data = result[0] if isinstance(result, list) else result

        # Raw emotion scores (they sum to ~100, like percentages)
        raw_scores = face_data["emotion"]          # e.g. {"happy": 68.3, "sad": 12.1, ...}

        # ── Step 3: Normalise to 0.0–1.0 probabilities ───────────────────────
        total = sum(raw_scores.values())
        normalised = {k: round(v / total, 4) for k, v in raw_scores.items()}

        # ── Step 4: Get top emotion ───────────────────────────────────────────
        top_emotion, confidence = get_top_label(normalised)

        return {
            "scores":        normalised,
            "top_emotion":   top_emotion,
            "confidence":    confidence,
            "display_label": EMOTION_LABEL_MAP.get(top_emotion, top_emotion.title()),
            "valence":       EMOTION_VALENCE_MAP.get(top_emotion, "neutral"),
            "error":         None,
        }

    except Exception as e:
        # If DeepFace fails (e.g. no face found), return a safe fallback
        print(f"[EmotionDetector] Error: {e}")
        return {
            "scores":        {"neutral": 1.0},
            "top_emotion":   "neutral",
            "confidence":    1.0,
            "display_label": "Neutral 😐 (face not detected)",
            "valence":       "neutral",
            "error":         str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run this file directly to verify it works:
#   python models/emotion_detector.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Use a test image if provided, otherwise use a placeholder
    test_image_path = sys.argv[1] if len(sys.argv) > 1 else None

    if test_image_path and Path(test_image_path).exists():
        print(f"\n🔍 Testing emotion detector on: {test_image_path}")
        img = Image.open(test_image_path)
    else:
        # Create a simple grey placeholder image for testing
        print("\n🔍 No image provided — using a blank test image (expect neutral/low confidence)")
        img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)

    result = analyse_emotion(img)

    print("\n📊 Results:")
    print(f"  Top Emotion   : {result['display_label']}")
    print(f"  Confidence    : {result['confidence']*100:.1f}%")
    print(f"  Valence       : {result['valence']}")
    print(f"  All Scores    :")
    for emotion, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 30)
        print(f"    {emotion:<10} {score*100:5.1f}%  {bar}")
    if result["error"]:
        print(f"\n  ⚠️  Warning: {result['error']}")
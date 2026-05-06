# ─────────────────────────────────────────────────────────────────────────────
# MoodSyncAI – models/fusion.py
#
# PURPOSE:
#   Combine the visual (face) and textual (sentiment) predictions into
#   a single unified assessment. Detect mismatches between modalities.
#
# HOW IT WORKS:
#   1. Takes emotion result  (from emotion_detector.py)
#   2. Takes sentiment result (from sentiment_analyzer.py)
#   3. Maps both to a common scale: positive / neutral / negative
#   4. Computes weighted average of their confidence scores
#   5. Detects MATCH or MISMATCH based on valence agreement
#   6. Returns a unified result dict ready for the UI and generator
# ─────────────────────────────────────────────────────────────────────────────

from config import (
    VISUAL_WEIGHT,
    TEXT_WEIGHT,
    MISMATCH_THRESHOLD,
    EMOTION_VALENCE_MAP,
)


# ── Valence scoring ───────────────────────────────────────────────────────────
# Convert valence label → numeric score for weighted averaging
VALENCE_SCORE = {
    "positive":  1.0,
    "neutral":   0.0,
    "negative": -1.0,
}


def fuse(emotion_result: dict, sentiment_result: dict) -> dict:
    """
    Fuse visual emotion and text sentiment into a unified assessment.

    Args:
        emotion_result   : dict from emotion_detector.analyse_emotion()
        sentiment_result : dict from sentiment_analyzer.analyse_sentiment()

    Returns a dict with:
        {
            "visual_valence":   "negative",
            "text_valence":     "positive",
            "fused_valence":    "neutral",
            "fused_score":      0.05,          ← -1.0 to +1.0 scale
            "is_mismatch":      True,
            "mismatch_degree":  0.75,          ← how different they are (0-1)
            "badge":            "⚠️ MISMATCH DETECTED",
            "badge_color":      "orange",
            "confidence":       0.84,          ← average confidence of both models
            "summary_context":  { ... }        ← rich context for the generator
        }
    """

    # ── Step 1: Get valence from each modality ────────────────────────────────
    visual_valence = emotion_result.get("valence", "neutral")
    text_valence   = sentiment_result.get("top_sentiment", "neutral")

    # Map text sentiment label to valence (they already match: positive/neutral/negative)
    # but emotion uses the EMOTION_VALENCE_MAP, which is already applied in emotion_detector

    # ── Step 2: Get confidence scores from each modality ─────────────────────
    visual_confidence = emotion_result.get("confidence", 0.5)
    text_confidence   = sentiment_result.get("confidence", 0.5)

    # Average confidence (how sure the system is overall)
    avg_confidence = round(
        (visual_confidence * VISUAL_WEIGHT) + (text_confidence * TEXT_WEIGHT), 4
    )

    # ── Step 3: Convert valences to numeric scores ────────────────────────────
    visual_score = VALENCE_SCORE.get(visual_valence, 0.0)
    text_score   = VALENCE_SCORE.get(text_valence, 0.0)

    # ── Step 4: Weighted average fusion ──────────────────────────────────────
    fused_score = round(
        (visual_score * VISUAL_WEIGHT) + (text_score * TEXT_WEIGHT), 4
    )

    # Convert fused score back to valence label
    if fused_score > 0.2:
        fused_valence = "positive"
    elif fused_score < -0.2:
        fused_valence = "negative"
    else:
        fused_valence = "neutral"

    # ── Step 5: Mismatch detection ────────────────────────────────────────────
    # Mismatch = face and text point in OPPOSITE directions
    mismatch_degree = abs(visual_score - text_score) / 2.0   # normalise to 0-1
    is_mismatch     = mismatch_degree >= MISMATCH_THRESHOLD

    # ── Step 6: Build badge for UI ────────────────────────────────────────────
    if is_mismatch:
        badge       = "⚠️ MISMATCH DETECTED"
        badge_color = "orange"
    else:
        badge       = "✅ SIGNALS ALIGNED"
        badge_color = "green"

    # ── Step 7: Build rich context dict for the generator ────────────────────
    summary_context = {
        "top_emotion":        emotion_result.get("top_emotion", "neutral"),
        "emotion_confidence": visual_confidence,
        "top_sentiment":      sentiment_result.get("top_sentiment", "neutral"),
        "sentiment_confidence": text_confidence,
        "visual_valence":     visual_valence,
        "text_valence":       text_valence,
        "fused_valence":      fused_valence,
        "is_mismatch":        is_mismatch,
        "mismatch_degree":    round(mismatch_degree, 4),
    }

    return {
        "visual_valence":  visual_valence,
        "text_valence":    text_valence,
        "fused_valence":   fused_valence,
        "fused_score":     fused_score,
        "is_mismatch":     is_mismatch,
        "mismatch_degree": round(mismatch_degree, 4),
        "badge":           badge,
        "badge_color":     badge_color,
        "confidence":      avg_confidence,
        "summary_context": summary_context,
    }


def describe_fusion(fusion_result: dict) -> str:
    """
    Return a short human-readable description of the fusion result.
    Used as a fallback if the generative model fails.

    Example outputs:
      "Face shows fear (negative), text is positive → MISMATCH"
      "Both face and text show positive signals → ALIGNED"
    """
    ctx = fusion_result["summary_context"]

    face_part = (
        f"Face shows {ctx['top_emotion']} "
        f"({ctx['visual_valence']}, {ctx['emotion_confidence']*100:.0f}%)"
    )
    text_part = (
        f"text is {ctx['top_sentiment']} "
        f"({ctx['sentiment_confidence']*100:.0f}%)"
    )

    if fusion_result["is_mismatch"]:
        return f"{face_part}, {text_part} → ⚠️ MISMATCH"
    else:
        return f"{face_part}, {text_part} → ✅ ALIGNED"


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
#   python models/fusion.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Simulate the professor's exact example:
    # Face → sad/fearful, Text → positive ("project is going really well")
    mock_emotion_result = {
        "top_emotion": "fear",
        "confidence":  0.68,
        "valence":     "negative",
        "scores":      {"fear": 0.68, "sad": 0.20, "neutral": 0.12},
    }

    mock_sentiment_result = {
        "top_sentiment": "positive",
        "confidence":    0.81,
        "scores":        {"positive": 0.81, "neutral": 0.12, "negative": 0.07},
    }

    result = fuse(mock_emotion_result, mock_sentiment_result)

    print("\n" + "="*60)
    print("  MoodSyncAI — Fusion Layer Test")
    print("="*60)
    print(f"\n  Visual Valence  : {result['visual_valence']}")
    print(f"  Text Valence    : {result['text_valence']}")
    print(f"  Fused Valence   : {result['fused_valence']}")
    print(f"  Fused Score     : {result['fused_score']:+.3f}  (-1=negative, +1=positive)")
    print(f"  Mismatch Degree : {result['mismatch_degree']*100:.1f}%")
    print(f"  Badge           : {result['badge']}")
    print(f"  Avg Confidence  : {result['confidence']*100:.1f}%")
    print(f"\n  Description     : {describe_fusion(result)}")

    # Test 2: Both positive (should be ALIGNED)
    print("\n" + "-"*60)
    print("  Test 2: Both signals positive")
    print("-"*60)
    mock_emotion_2 = {"top_emotion": "happy", "confidence": 0.90, "valence": "positive", "scores": {}}
    mock_sentiment_2 = {"top_sentiment": "positive", "confidence": 0.95, "scores": {}}
    result2 = fuse(mock_emotion_2, mock_sentiment_2)
    print(f"  Badge : {result2['badge']}")
    print(f"  Description: {describe_fusion(result2)}")
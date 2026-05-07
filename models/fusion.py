# ─────────────────────────────────────────────────────────────────────────────
# MoodSyncAI – models/fusion.py
# Combines rule-based fusion WITH a learned neural network fusion layer
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import numpy as np

from config import (
    VISUAL_WEIGHT,
    TEXT_WEIGHT,
    MISMATCH_THRESHOLD,
    EMOTION_VALENCE_MAP,
)

VALENCE_SCORE = {
    "positive":  1.0,
    "neutral":   0.0,
    "negative": -1.0,
}

# ── Learned Fusion Network ────────────────────────────────────────────────────
class FusionNet(nn.Module):
    """
    Small MLP that takes 7 emotion scores + 3 sentiment scores = 10 inputs
    and outputs 3 values: [positive, neutral, negative] probabilities.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


# Instantiate model once at module load
_fusion_model = FusionNet()
_fusion_model.eval()

EMOTION_ORDER    = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SENTIMENT_ORDER  = ["positive", "neutral", "negative"]


def _learned_fusion(emotion_scores: dict, sentiment_scores: dict) -> dict:
    """
    Run the learned fusion network.
    Returns {"valence": str, "confidence": float, "scores": dict}
    """
    try:
        e_vec = [emotion_scores.get(e, 0.0) for e in EMOTION_ORDER]
        s_vec = [sentiment_scores.get(s, 0.0) for s in SENTIMENT_ORDER]
        x = torch.tensor(e_vec + s_vec, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            probs = _fusion_model(x).squeeze(0).numpy()

        labels = ["positive", "neutral", "negative"]
        idx    = int(np.argmax(probs))
        return {
            "valence":    labels[idx],
            "confidence": float(probs[idx]),
            "scores":     dict(zip(labels, probs.tolist())),
        }
    except Exception as e:
        print(f"[LearnedFusion] Error: {e} — falling back to rule-based")
        return None


def fuse(emotion_result: dict, sentiment_result: dict) -> dict:
    """Fuse visual emotion and text sentiment into a unified assessment."""

    visual_valence    = emotion_result.get("valence", "neutral")
    text_valence      = sentiment_result.get("top_sentiment", "neutral")
    visual_confidence = emotion_result.get("confidence", 0.5)
    text_confidence   = sentiment_result.get("confidence", 0.5)

    avg_confidence = round(
        (visual_confidence * VISUAL_WEIGHT) + (text_confidence * TEXT_WEIGHT), 4
    )

    visual_score = VALENCE_SCORE.get(visual_valence, 0.0)
    text_score   = VALENCE_SCORE.get(text_valence, 0.0)

    # ── Try learned fusion first ──────────────────────────────────────────────
    learned = _learned_fusion(
        emotion_result.get("scores", {}),
        sentiment_result.get("scores", {}),
    )

    if learned:
        fused_valence    = learned["valence"]
        fused_score      = VALENCE_SCORE.get(fused_valence, 0.0)
        fusion_method    = "neural"
    else:
        # Fallback: weighted average
        fused_score = round(
            (visual_score * VISUAL_WEIGHT) + (text_score * TEXT_WEIGHT), 4
        )
        if fused_score > 0.2:
            fused_valence = "positive"
        elif fused_score < -0.2:
            fused_valence = "negative"
        else:
            fused_valence = "neutral"
        fusion_method = "rule-based"

    # ── Mismatch detection ────────────────────────────────────────────────────
    mismatch_degree = abs(visual_score - text_score) / 2.0
    is_mismatch     = mismatch_degree >= MISMATCH_THRESHOLD

    badge       = "⚠️ MISMATCH DETECTED" if is_mismatch else "✅ SIGNALS ALIGNED"
    badge_color = "orange"               if is_mismatch else "green"

    summary_context = {
        "top_emotion":          emotion_result.get("top_emotion", "neutral"),
        "emotion_confidence":   visual_confidence,
        "top_sentiment":        sentiment_result.get("top_sentiment", "neutral"),
        "sentiment_confidence": text_confidence,
        "visual_valence":       visual_valence,
        "text_valence":         text_valence,
        "fused_valence":        fused_valence,
        "is_mismatch":          is_mismatch,
        "mismatch_degree":      round(mismatch_degree, 4),
        "fusion_method":        fusion_method,
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
        "fusion_method":   fusion_method,
        "summary_context": summary_context,
    }


def describe_fusion(fusion_result: dict) -> str:
    ctx = fusion_result["summary_context"]
    face_part = (
        f"Face shows {ctx['top_emotion']} "
        f"({ctx['visual_valence']}, {ctx['emotion_confidence']*100:.0f}%)"
    )
    text_part = (
        f"text is {ctx['top_sentiment']} "
        f"({ctx['sentiment_confidence']*100:.0f}%)"
    )
    method = ctx.get("fusion_method", "rule-based")
    tag    = "⚠️ MISMATCH" if fusion_result["is_mismatch"] else "✅ ALIGNED"
    return f"{face_part}, {text_part} → {tag} [{method} fusion]"


if __name__ == "__main__":
    mock_emotion = {
        "top_emotion": "fear", "confidence": 0.68, "valence": "negative",
        "scores": {"angry": 0.05, "disgust": 0.02, "fear": 0.68,
                   "happy": 0.02, "neutral": 0.12, "sad": 0.10, "surprise": 0.01},
    }
    mock_sentiment = {
        "top_sentiment": "positive", "confidence": 0.81,
        "scores": {"positive": 0.81, "neutral": 0.12, "negative": 0.07},
    }
    result = fuse(mock_emotion, mock_sentiment)
    print(f"Badge: {result['badge']}")
    print(f"Fusion method: {result['fusion_method']}")
    print(f"Description: {describe_fusion(result)}")
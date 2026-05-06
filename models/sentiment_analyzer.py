# ─────────────────────────────────────────────────────────────────────────────
# MoodSyncAI – models/sentiment_analyzer.py
#
# PURPOSE:
#   Analyse sentiment from text using a pretrained RoBERTa transformer.
#   This is the TEXT branch of our multimodal system.
#
# MODEL USED:
#   cardiffnlp/twitter-roberta-base-sentiment-latest
#   - Trained on millions of tweets
#   - Classifies into: NEGATIVE / NEUTRAL / POSITIVE
#   - Downloads automatically on first run (~500MB, cached after)
#
# HOW IT WORKS:
#   1. Takes a string of text
#   2. Tokenises it (converts words → numbers the model understands)
#   3. RoBERTa transformer processes it → returns raw logits
#   4. Softmax converts logits → probabilities (0.0 – 1.0)
#   5. Returns clean dict + top sentiment + confidence
# ─────────────────────────────────────────────────────────────────────────────

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

from config import TEXT_MODEL_NAME, SENTIMENT_LABEL_MAP
from utils.helpers import get_top_label

# ── Module-level cache ────────────────────────────────────────────────────────
# We load the model ONCE and reuse it — avoids reloading on every call
_tokenizer = None
_model     = None


def _load_model():
    """
    Load the RoBERTa tokenizer and model.
    Called automatically on first use — cached after that.
    """
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        print(f"[SentimentAnalyzer] Loading model: {TEXT_MODEL_NAME}")
        print("[SentimentAnalyzer] First run may take 1-2 mins (downloading ~500MB)...")

        _tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        _model     = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME)
        _model.eval()   # set to evaluation mode (disables dropout etc.)

        print("[SentimentAnalyzer] ✅ Model loaded successfully!")

    return _tokenizer, _model


def analyse_sentiment(text: str) -> dict:
    """
    Analyse sentiment of the given text string.

    Args:
        text : str — the sentence/paragraph to analyse

    Returns a dict with:
        {
            "scores":        {"positive": 0.81, "neutral": 0.12, "negative": 0.07},
            "top_sentiment": "positive",
            "confidence":    0.81,
            "display_label": "Positive 😊",
            "error":         None
        }
    """

    # ── Guard: empty text ─────────────────────────────────────────────────────
    if not text or not text.strip():
        return {
            "scores":        {"neutral": 1.0, "positive": 0.0, "negative": 0.0},
            "top_sentiment": "neutral",
            "confidence":    1.0,
            "display_label": "Neutral 😐 (no text provided)",
            "error":         "Empty text input",
        }

    try:
        # ── Step 1: Load model (cached after first call) ──────────────────────
        tokenizer, model = _load_model()

        # ── Step 2: Tokenise the text ─────────────────────────────────────────
        # truncation=True  → cuts text longer than 512 tokens (model limit)
        # return_tensors   → returns PyTorch tensors
        inputs = tokenizer(
            text,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 512,
            padding        = True,
        )

        # ── Step 3: Run through RoBERTa ───────────────────────────────────────
        with torch.no_grad():            # no_grad = faster, less memory (we're not training)
            outputs = model(**inputs)
            logits  = outputs.logits     # raw scores before softmax

        # ── Step 4: Convert to probabilities ──────────────────────────────────
        probs = F.softmax(logits, dim=-1).squeeze().tolist()

        # ── Step 5: Map to label names ────────────────────────────────────────
        # RoBERTa model labels are stored in model.config.id2label
        # Typically: {0: "negative", 1: "neutral", 2: "positive"}
        id2label = model.config.id2label
        scores   = {
            id2label[i].lower(): round(float(p), 4)
            for i, p in enumerate(probs)
        }

        # ── Step 6: Get top sentiment ─────────────────────────────────────────
        top_sentiment, confidence = get_top_label(scores)

        return {
            "scores":        scores,
            "top_sentiment": top_sentiment,
            "confidence":    confidence,
            "display_label": SENTIMENT_LABEL_MAP.get(top_sentiment, top_sentiment.title()),
            "error":         None,
        }

    except Exception as e:
        print(f"[SentimentAnalyzer] Error: {e}")
        return {
            "scores":        {"neutral": 1.0, "positive": 0.0, "negative": 0.0},
            "top_sentiment": "neutral",
            "confidence":    1.0,
            "display_label": "Neutral 😐 (analysis failed)",
            "error":         str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run this file directly to verify it works:
#   python models/sentiment_analyzer.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    test_sentences = [
        "No, I think the project is going really well.",   # positive (professor's example)
        "I am absolutely devastated and hopeless.",        # negative
        "The meeting is at 3pm tomorrow.",                 # neutral
        "This is the best day of my life!",               # positive
        "I hate everything about this situation.",         # negative
    ]

    print("\n" + "="*60)
    print("  MoodSyncAI — Sentiment Analyser Test")
    print("="*60)

    for sentence in test_sentences:
        result = analyse_sentiment(sentence)
        print(f"\n📝 Text      : {sentence}")
        print(f"   Sentiment : {result['display_label']}")
        print(f"   Confidence: {result['confidence']*100:.1f}%")
        print(f"   All Scores:")
        for label, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 30)
            print(f"     {label:<10} {score*100:5.1f}%  {bar}")
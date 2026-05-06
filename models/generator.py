# ─────────────────────────────────────────────────────────────────────────────
# MoodSyncAI – models/generator.py
#
# PURPOSE:
#   Generate a natural language summary of the combined emotional state.
#   This is the GENERATIVE component required by the assignment.
#
# HOW IT WORKS:
#   1. Takes the fusion result (mismatch/aligned + all scores)
#   2. Builds a smart prompt describing what the models found
#   3. Feeds it into GPT-2 to generate a plain English summary
#   4. Falls back to a rule-based summary if GPT-2 output is poor
# ─────────────────────────────────────────────────────────────────────────────

import re
import torch
from transformers import pipeline, set_seed

from config import (
    GENERATOR_MODEL,
    GENERATOR_MAX_NEW_TOKENS,
    GENERATOR_TEMPERATURE,
)

# ── Module-level cache ────────────────────────────────────────────────────────
_generator = None


def _load_generator():
    """Load GPT-2 text generation pipeline (cached after first call)."""
    global _generator
    if _generator is None:
        print(f"[Generator] Loading model: {GENERATOR_MODEL}")
        print("[Generator] First run downloads ~500MB — please wait...")
        _generator = pipeline(
            "text-generation",
            model     = GENERATOR_MODEL,
            device    = -1,       # -1 = CPU (works on all machines)
        )
        set_seed(42)              # reproducible outputs
        print("[Generator] ✅ Model loaded!")
    return _generator


def _build_prompt(fusion_result: dict) -> str:
    """
    Build a clear, structured prompt for GPT-2 based on fusion results.
    The better the prompt, the better the generated summary.
    """
    ctx = fusion_result["summary_context"]

    emotion    = ctx["top_emotion"]
    emo_conf   = int(ctx["emotion_confidence"] * 100)
    sentiment  = ctx["top_sentiment"]
    sent_conf  = int(ctx["sentiment_confidence"] * 100)
    is_mismatch = ctx["is_mismatch"]

    if is_mismatch:
        prompt = (
            f"Emotional analysis report: The person's face shows {emotion} "
            f"with {emo_conf}% confidence, indicating {ctx['visual_valence']} emotions. "
            f"However, their words express {sentiment} sentiment at {sent_conf}% confidence. "
            f"This is a significant emotional mismatch. "
            f"Professional assessment:"
        )
    else:
        prompt = (
            f"Emotional analysis report: The person's face shows {emotion} "
            f"with {emo_conf}% confidence. "
            f"Their words also express {sentiment} sentiment at {sent_conf}% confidence. "
            f"Both signals are aligned. "
            f"Professional assessment:"
        )

    return prompt


def _rule_based_summary(fusion_result: dict) -> str:
    """
    Fallback rule-based summary — always clean and professional.
    Used when GPT-2 generates poor output.
    """
    ctx         = fusion_result["summary_context"]
    emotion     = ctx["top_emotion"]
    emo_conf    = int(ctx["emotion_confidence"] * 100)
    sentiment   = ctx["top_sentiment"]
    sent_conf   = int(ctx["sentiment_confidence"] * 100)
    is_mismatch = ctx["is_mismatch"]

    if is_mismatch:
        # Specific mismatch messages based on combination
        if ctx["visual_valence"] == "negative" and ctx["text_valence"] == "positive":
            return (
                f"Despite expressing {sentiment} sentiment verbally ({sent_conf}% confidence), "
                f"the speaker's facial cues indicate {emotion} ({emo_conf}% confidence). "
                f"This incongruence between words and expression is worth noting — "
                f"the person may be masking their true emotional state."
            )
        elif ctx["visual_valence"] == "positive" and ctx["text_valence"] == "negative":
            return (
                f"The speaker's words convey {sentiment} sentiment ({sent_conf}% confidence), "
                f"yet their facial expression shows {emotion} ({emo_conf}% confidence). "
                f"This mismatch may suggest forced positivity or emotional conflict."
            )
        else:
            return (
                f"A mismatch was detected between the visual signal ({emotion}, {emo_conf}%) "
                f"and textual signal ({sentiment}, {sent_conf}%). "
                f"The person's expressed emotions do not fully align with their words."
            )
    else:
        # Aligned messages
        if ctx["fused_valence"] == "positive":
            return (
                f"Both facial expression ({emotion}, {emo_conf}%) and verbal sentiment "
                f"({sentiment}, {sent_conf}%) indicate a positive emotional state. "
                f"The person appears genuinely content and their words reflect this."
            )
        elif ctx["fused_valence"] == "negative":
            return (
                f"Both facial expression ({emotion}, {emo_conf}%) and verbal sentiment "
                f"({sentiment}, {sent_conf}%) consistently indicate distress. "
                f"The person's words and expression are aligned in conveying negative emotions."
            )
        else:
            return (
                f"The person displays a neutral emotional state across both modalities — "
                f"face shows {emotion} ({emo_conf}%) and speech is {sentiment} ({sent_conf}%). "
                f"No strong emotional signal detected."
            )


def _clean_gpt2_output(generated: str, prompt: str) -> str:
    """
    Clean up GPT-2 output — remove the prompt, fix whitespace,
    and return only the generated part.
    """
    # Remove the prompt from the beginning
    summary = generated[len(prompt):].strip()

    # Remove incomplete last sentence (no ending punctuation)
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    complete  = [s for s in sentences if s and s[-1] in ".!?"]
    summary   = " ".join(complete[:3])   # max 3 sentences

    return summary.strip()


def generate_summary(fusion_result: dict) -> str:
    """
    Generate a natural language summary of the emotional analysis.

    Strategy:
      - Primary  : rule-based summary (always clean and professional)
      - Secondary: GPT-2 enrichment appended if it produces sensible output

    Args:
        fusion_result : dict from fusion.fuse()

    Returns:
        str — a 2-3 sentence plain English summary
    """
    # ── Always start with the clean rule-based summary ────────────────────────
    base_summary = _rule_based_summary(fusion_result)

    try:
        gen    = _load_generator()
        prompt = _build_prompt(fusion_result)

        outputs = gen(
            prompt,
            max_new_tokens     = 60,
            temperature        = 0.6,
            do_sample          = True,
            top_p              = 0.90,
            repetition_penalty = 1.4,
            num_return_sequences = 1,
            pad_token_id       = 50256,
        )

        generated_text = outputs[0]["generated_text"]
        gpt2_part      = _clean_gpt2_output(generated_text, prompt)

        # Only use GPT-2 addition if it looks like plain English
        # (no brackets, citations, or medical jargon signals)
        bad_signals = ["[", "eTable", "Table 1", "et al", "p <", "Figure", "We assessed", "patient by"]
        is_clean    = (
            len(gpt2_part.split()) >= 10
            and not any(sig in gpt2_part for sig in bad_signals)
        )

        if is_clean:
            return base_summary + " " + gpt2_part

        return base_summary

    except Exception as e:
        print(f"[Generator] GPT-2 skipped: {e}")
        return base_summary


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
#   python models/generator.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Test 1: Mismatch (professor's example)
    mismatch_fusion = {
        "is_mismatch":     True,
        "mismatch_degree": 1.0,
        "badge":           "⚠️ MISMATCH DETECTED",
        "fused_valence":   "neutral",
        "summary_context": {
            "top_emotion":          "fear",
            "emotion_confidence":   0.68,
            "top_sentiment":        "positive",
            "sentiment_confidence": 0.81,
            "visual_valence":       "negative",
            "text_valence":         "positive",
            "fused_valence":        "neutral",
            "is_mismatch":          True,
            "mismatch_degree":      1.0,
        },
    }

    # Test 2: Aligned positive
    aligned_fusion = {
        "is_mismatch":     False,
        "mismatch_degree": 0.0,
        "badge":           "✅ SIGNALS ALIGNED",
        "fused_valence":   "positive",
        "summary_context": {
            "top_emotion":          "happy",
            "emotion_confidence":   0.90,
            "top_sentiment":        "positive",
            "sentiment_confidence": 0.95,
            "visual_valence":       "positive",
            "text_valence":         "positive",
            "fused_valence":        "positive",
            "is_mismatch":          False,
            "mismatch_degree":      0.0,
        },
    }

    print("\n" + "="*60)
    print("  MoodSyncAI — Generator Test")
    print("="*60)

    print("\n📋 Test 1: MISMATCH (face=fearful, text=positive)")
    print("-"*60)
    summary1 = generate_summary(mismatch_fusion)
    print(f"  Summary: {summary1}")

    print("\n📋 Test 2: ALIGNED (face=happy, text=positive)")
    print("-"*60)
    summary2 = generate_summary(aligned_fusion)
    print(f"  Summary: {summary2}")
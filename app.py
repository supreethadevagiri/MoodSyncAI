# ─────────────────────────────────────────────────────────────────────────────
# MoodSyncAI – app.py
# Run with: python app.py
# ─────────────────────────────────────────────────────────────────────────────

import gradio as gr
from PIL import Image

from models.emotion_detector import analyse_emotion
from models.sentiment_analyzer import analyse_sentiment
from models.fusion import fuse, describe_fusion
from models.generator import generate_summary
from models.audio_transcriber import transcribe_audio
from utils.helpers import make_bar_chart, format_confidence
from utils.gradcam import generate_gradcam
from config import APP_TITLE, APP_DESCRIPTION

# ── In-memory history (last 3 analyses) ──────────────────────────────────────
_history: list[dict] = []


def analyse(image, text, audio):
    """
    Full multimodal analysis pipeline.
    Supports 3 input modes:
      - image + text         (original 2-modal)
      - image + audio        (Whisper transcribes → text, then same pipeline)
      - image + text + audio (text takes priority; transcript shown separately)
    """

    if image is None:
        msg = "Please upload a face photo."
        return None, msg, None, msg, msg, msg, None, _render_history(), ""

    # ── Audio transcription (if audio provided and text box is empty) ─────────
    transcript_display = ""
    if audio is not None:
        transcription = transcribe_audio(audio)
        if transcription["error"] is None and transcription["transcript"]:
            transcript_display = transcription["transcript"]
            # Only override text if the user left the text box empty
            if not text or not text.strip():
                text = transcript_display

    if not text or not text.strip():
        msg = "Please type text OR upload an audio clip."
        return None, msg, None, msg, msg, msg, None, _render_history(), ""

    # Step 1: Visual emotion
    emotion_result = analyse_emotion(image)
    emotion_chart = make_bar_chart(
        emotion_result["scores"],
        title="Visual Emotion Scores",
        color="#E07B54",
    )
    emotion_label = (
        f"{emotion_result['display_label']}  —  "
        f"{format_confidence(emotion_result['confidence'])}"
    )

    # Step 2: Text sentiment
    sentiment_result = analyse_sentiment(text)
    sentiment_chart = make_bar_chart(
        sentiment_result["scores"],
        title="Textual Sentiment Scores",
        color="#4F86C6",
    )
    sentiment_label = (
        f"{sentiment_result['display_label']}  —  "
        f"{format_confidence(sentiment_result['confidence'])}"
    )

    # Step 3: Fusion
    fusion_result = fuse(emotion_result, sentiment_result)
    badge = fusion_result["badge"]

    # Fusion method tag
    method = fusion_result.get("method", "rule")
    method_tag = "🧠 Neural Fusion" if method == "neural" else "📐 Rule-based Fusion"
    badge_with_method = f"{badge}     [{method_tag}]"

    # Step 4: Summary
    summary = generate_summary(fusion_result)

    # Step 5: Grad-CAM
    gradcam_image = generate_gradcam(image, emotion_result["top_emotion"])

    # Step 6: Update history
    source = "🎙️ audio" if (audio is not None and transcript_display) else "✍️ text"
    _history.insert(0, {
        "text":      text[:40] + ("…" if len(text) > 40 else ""),
        "source":    source,
        "emotion":   emotion_result["display_label"],
        "sentiment": sentiment_result["display_label"],
        "result":    "⚠️ Mismatch" if "MISMATCH" in badge else "✅ Aligned",
    })
    if len(_history) > 3:
        _history.pop()

    return (
        emotion_chart,
        emotion_label,
        sentiment_chart,
        sentiment_label,
        badge_with_method,
        summary,
        gradcam_image,
        _render_history(),
        transcript_display,
    )


def _render_history() -> str:
    """Render the last 3 analyses as a markdown table."""
    if not _history:
        return "_No analyses yet. Run your first analysis above!_"
    rows = ["| # | Text | Source | Emotion | Sentiment | Result |",
            "|---|------|--------|---------|-----------|--------|"]
    for i, h in enumerate(_history, 1):
        rows.append(
            f"| {i} | {h['text']} | {h['source']} | {h['emotion']} | {h['sentiment']} | {h['result']} |"
        )
    return "\n".join(rows)


# ── Example presets ───────────────────────────────────────────────────────────
EXAMPLES = [
    {
        "label": "😊 Happy text",
        "text": "I am so happy today, everything is going great!",
    },
    {
        "label": "😤 Dismissive colleague",
        "text": "No, I think the project is going really well.",
    },
    {
        "label": "😰 Anxious reassurance",
        "text": "Everything is totally fine, I am not stressed at all.",
    },
]

# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
}

.mood-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.08);
}
.mood-header h1 {
    font-size: 2.2rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.5px;
}
.mood-header p {
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
    line-height: 1.5;
}

.section-label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
    margin-bottom: 10px !important;
}

.example-btn {
    font-size: 0.82rem !important;
    padding: 6px 14px !important;
    border-radius: 20px !important;
    border: 1px solid #4f46e5 !important;
    background: #eef2ff !important;
    color: #3730a3 !important;
    cursor: pointer;
    transition: all 0.15s ease;
}
.example-btn:hover {
    background: #c7d2fe !important;
    border-color: #4f46e5 !important;
    color: #1e1b4b !important;
}

button.lg.primary {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
button.lg.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
}

.divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 8px 0;
}

textarea[data-testid="textbox"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
}

.history-section table {
    font-size: 0.85rem !important;
    width: 100%;
}

.audio-info {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #166534;
    margin-top: 6px;
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="MoodSyncAI", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:

    # ── Header ───────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="mood-header">
        <h1>🎭 MoodSyncAI</h1>
        <p>
            Multi-modal emotion &amp; sentiment analyser &nbsp;·&nbsp;
            Upload a face photo + type text <em>or</em> upload an audio clip
            to detect hidden emotional mismatches.
        </p>
    </div>
    """)

    gr.HTML('<hr class="divider">')

    # ── Input row ─────────────────────────────────────────────────────────────
    with gr.Row():

        # Left column — face photo
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📷  Upload Face Photo",
                height=300,
            )

        # Right column — text OR audio (tabs)
        with gr.Column(scale=1):
            with gr.Tabs():

                # Tab 1: typed text (original mode)
                with gr.Tab("✍️  Type Text"):
                    text_input = gr.Textbox(
                        label="💬  What did the person say?",
                        placeholder="Type the sentence here…",
                        lines=4,
                    )
                    gr.HTML('<p class="section-label" style="margin-top:8px">⚡ Quick examples</p>')
                    with gr.Row():
                        for ex in EXAMPLES:
                            gr.Button(
                                ex["label"],
                                elem_classes=["example-btn"],
                                size="sm",
                            ).click(
                                fn=lambda t=ex["text"]: t,
                                inputs=[],
                                outputs=[text_input],
                                queue=False,
                            )

                # Tab 2: audio upload (new 3rd modality)
                with gr.Tab("🎙️  Upload Audio"):
                    audio_input = gr.Audio(
                        type="filepath",
                        label="Upload a short audio clip (wav / mp3 / m4a)",
                    )
                    gr.HTML("""
                    <div class="audio-info">
                        🎙️ <strong>How it works:</strong> Whisper AI automatically transcribes
                        what is said in the clip. The transcript feeds into the same
                        sentiment + fusion pipeline as typed text — giving you a
                        <strong>3-modality analysis</strong> (face + voice + text).
                        Max 30 seconds recommended.
                    </div>
                    """)
                    transcript_out = gr.Textbox(
                        label="📝 Auto-transcript (what Whisper heard)",
                        interactive=False,
                        lines=3,
                        placeholder="Transcript appears here after analysis…",
                    )

            analyse_btn = gr.Button("🔍  Analyse", variant="primary", size="lg")

    gr.HTML('<hr class="divider">')

    # ── Results ───────────────────────────────────────────────────────────────
    gr.HTML('<p class="section-label">📊 Analysis results</p>')

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 👁️ Visual Emotion")
            emotion_chart_out = gr.Image(label="Emotion Confidence Chart", height=260)
            emotion_label_out = gr.Textbox(label="Top Emotion", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("#### 💬 Textual Sentiment")
            sentiment_chart_out = gr.Image(label="Sentiment Confidence Chart", height=260)
            sentiment_label_out = gr.Textbox(label="Top Sentiment", interactive=False)

    gr.HTML('<hr class="divider">')

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 🔀 Fusion Result")
            badge_out = gr.Textbox(
                label="Mismatch Detection + Fusion Method",
                interactive=False,
                lines=2,
            )
        with gr.Column(scale=2):
            gr.Markdown("#### 📝 Generative Summary")
            summary_out = gr.Textbox(
                label="Emotional Assessment",
                lines=4,
                interactive=False,
            )

    gr.HTML('<hr class="divider">')

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 🔥 Grad-CAM Attention Map")
            gradcam_out = gr.Image(
                label="Face regions that influenced the emotion prediction",
                height=300,
            )

    gr.HTML('<hr class="divider">')

    # ── History ───────────────────────────────────────────────────────────────
    gr.HTML('<p class="section-label">🕒 Recent analyses (this session)</p>')
    history_out = gr.Markdown(
        value=_render_history(),
        elem_classes=["history-section"],
    )

    # ── Wire button ───────────────────────────────────────────────────────────
    analyse_btn.click(
        fn=analyse,
        inputs=[image_input, text_input, audio_input],
        outputs=[
            emotion_chart_out,
            emotion_label_out,
            sentiment_chart_out,
            sentiment_label_out,
            badge_out,
            summary_out,
            gradcam_out,
            history_out,
            transcript_out,
        ],
    )


if __name__ == "__main__":
    demo.launch(share=True, debug=True)
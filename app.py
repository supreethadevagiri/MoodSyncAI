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

    transcript_display = ""
    if audio is not None:
        transcription = transcribe_audio(audio)
        if transcription["error"] is None and transcription["transcript"]:
            transcript_display = transcription["transcript"]
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
    if not _history:
        return "_No analyses yet. Run your first analysis above!_"
    rows = ["| # | Text | Source | Emotion | Sentiment | Result |",
            "|---|------|--------|---------|-----------|--------|"]
    for i, h in enumerate(_history, 1):
        rows.append(
            f"| {i} | {h['text']} | {h['source']} | {h['emotion']} | {h['sentiment']} | {h['result']} |"
        )
    return "\n".join(rows)


EXAMPLES = [
    {"label": "😊 Happy text",        "text": "I am so happy today, everything is going great!"},
    {"label": "😤 Dismissive colleague","text": "No, I think the project is going really well."},
    {"label": "😰 Anxious reassurance", "text": "Everything is totally fine, I am not stressed at all."},
]

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Syne', sans-serif !important;
    background: #07080d !important;
    color: #ffffff !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 0 24px 60px !important;
}

.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 10%, rgba(91,33,182,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 80%, rgba(6,182,212,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(236,72,153,0.06) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ── Header ── */
.msai-header {
    position: relative;
    padding: 56px 0 40px;
    text-align: center;
}

.msai-header-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #7c3aed;
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 100px;
    padding: 6px 16px;
    margin-bottom: 22px;
    animation: fadeSlideDown 0.6s ease both;
}

.msai-header-eyebrow::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #7c3aed;
    box-shadow: 0 0 8px #7c3aed;
    animation: pulse 2s ease infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.85); }
}

.msai-title {
    font-size: clamp(2.8rem, 6vw, 5rem) !important;
    font-weight: 800 !important;
    line-height: 1.05 !important;
    letter-spacing: -0.03em !important;
    margin: 0 0 18px !important;
    background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 40%, #67e8f9 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    animation: fadeSlideDown 0.7s 0.1s ease both;
}

.msai-subtitle {
    font-size: 1rem !important;
    color: #a1a1aa !important;
    font-weight: 400 !important;
    max-width: 520px;
    margin: 0 auto !important;
    line-height: 1.7 !important;
    animation: fadeSlideDown 0.7s 0.2s ease both;
}

.msai-subtitle strong { color: #a78bfa; font-weight: 600; }

@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Divider ── */
.msai-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,58,237,0.3), transparent);
    margin: 8px 0 32px;
}

/* ── Section labels (01 / Input etc.) ── */
.msai-section-label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #71717a !important;
    margin: 0 0 20px !important;
    display: flex;
    align-items: center;
    gap: 10px;
}

.msai-section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(113,113,122,0.4), transparent);
}

/* ── Markdown section headers (#### 👁️ Visual Emotion etc.) ── */
.gradio-container .prose h4,
.gradio-container .md h4,
.gradio-container h4 {
    font-family:    'Syne', sans-serif !important;
    font-weight:    800 !important;
    font-size:      1.1rem !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
    color:          #ffffff !important;
    margin:         0 0 16px !important;
    padding:        0 !important;
    background:     transparent !important;
    border:         none !important;
    line-height:    1.3 !important;
}

/* ── Inputs ── */
.gradio-container input,
.gradio-container textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: rgba(124,58,237,0.5) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.1) !important;
    outline: none !important;
}

/* ── Image upload ── */
.gradio-container .image-container,
.gradio-container [data-testid="image"] {
    border: 2px dashed rgba(124,58,237,0.25) !important;
    border-radius: 16px !important;
    background: rgba(124,58,237,0.03) !important;
    transition: border-color 0.2s ease !important;
}

.gradio-container .image-container:hover,
.gradio-container [data-testid="image"]:hover {
    border-color: rgba(124,58,237,0.5) !important;
}

/* ── Tabs ── */
.gradio-container .tab-nav {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    gap: 4px !important;
}

.gradio-container .tab-nav button {
    border-radius: 9px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.02em !important;
    color: #a1a1aa !important;
    transition: all 0.2s ease !important;
    padding: 8px 18px !important;
    border: none !important;
    background: transparent !important;
}

.gradio-container .tab-nav button.selected {
    background: linear-gradient(135deg, #7c3aed, #0891b2) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 12px rgba(124,58,237,0.35) !important;
}

/* ── Analyse Button ── */
button.lg.primary,
.gradio-container button[variant="primary"] {
    background: linear-gradient(135deg, #7c3aed 0%, #0891b2 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: #ffffff !important;
    padding: 14px 36px !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}

button.lg.primary:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 32px rgba(124,58,237,0.55) !important;
}

button.lg.primary:active {
    transform: translateY(0) scale(0.99) !important;
}

/* ── Example pills ── */
.example-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem !important;
    padding: 7px 16px !important;
    border-radius: 100px !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    background: rgba(124,58,237,0.06) !important;
    color: #a78bfa !important;
    cursor: pointer;
    transition: all 0.18s ease;
    white-space: nowrap;
}

.example-pill:hover {
    background: rgba(124,58,237,0.15) !important;
    border-color: rgba(124,58,237,0.6) !important;
    color: #c4b5fd !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(124,58,237,0.2);
}

/* ══════════════════════════════════════════════════════
   LABEL FIX
   Strip the Gradio 5 purple pill from every field label.
   Field labels (the small ones under each widget) → white mono text.
   Does NOT touch h4 headers (handled separately above).
══════════════════════════════════════════════════════ */
.gradio-container .label-wrap,
.gradio-container .label-wrap * {
    background:       transparent !important;
    background-color: transparent !important;
    border:           none !important;
    box-shadow:       none !important;
    border-radius:    0 !important;
    padding:          0 !important;
    margin:           0 !important;
}

.gradio-container .label-wrap span,
.gradio-container .label-wrap > span {
    color:          #ffffff !important;
    font-family:    'DM Mono', monospace !important;
    font-size:      0.68rem !important;
    font-weight:    400 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    display:        block !important;
    margin-bottom:  4px !important;
}

/* ── Read-only result textboxes ── */
.gradio-container textarea[readonly],
.gradio-container input[readonly] {
    background:   rgba(124,58,237,0.04) !important;
    border-color: rgba(124,58,237,0.12) !important;
    color:        #e2e8f0 !important;
    font-family:  'DM Mono', monospace !important;
}

/* ── Output images ── */
.gradio-container .output-image img {
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── History table ── */
.history-wrap table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}

.history-wrap table th {
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.65rem !important;
    color: #71717a;
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    text-align: left;
}

.history-wrap table td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    color: #e4e4e7;
}

.history-wrap table tr:last-child td { border-bottom: none; }
.history-wrap table tr:hover td { background: rgba(124,58,237,0.04); }

/* ── Audio callout ── */
.audio-callout {
    background: rgba(6,182,212,0.06);
    border: 1px solid rgba(6,182,212,0.18);
    border-radius: 12px;
    padding: 14px 18px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #67e8f9;
    line-height: 1.6;
    margin-top: 10px;
}

.audio-callout strong { color: #a5f3fc; font-weight: 500; }

/* ── Fusion badge ── */
.badge-glow textarea {
    background:   rgba(124,58,237,0.06) !important;
    border-color: rgba(124,58,237,0.25) !important;
    color:        #c4b5fd !important;
    font-size:    0.95rem !important;
    font-weight:  600 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.5); }

/* ── Gradio quirks ── */
.gradio-container .wrap { gap: 0 !important; }
.gradio-container footer { display: none !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="MoodSyncAI", theme=gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="cyan",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Syne"), "sans-serif"],
), css=CUSTOM_CSS) as demo:

    # ── Header ───────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="msai-header">
        <div class="msai-header-eyebrow">Multimodal Intelligence · v2.0</div>
        <h1 class="msai-title">MoodSync AI</h1>
        <p class="msai-subtitle">
            Detect hidden emotional mismatches across <strong>face&nbsp;·&nbsp;text&nbsp;·&nbsp;voice</strong>.
            Upload a photo, add what was said, and let the model reveal what the numbers know.
        </p>
    </div>
    """)

    gr.HTML('<div class="msai-divider"></div>')

    # ── Input ─────────────────────────────────────────────────────────────────
    gr.HTML('<p class="msai-section-label">01 &nbsp;/&nbsp; Input</p>')

    with gr.Row(equal_height=True):

        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Face Photo or Use Webcam",
                height=310,
                sources=["upload", "webcam"],
            )

        with gr.Column(scale=1):
            with gr.Tabs():

                with gr.Tab("✍️  Type Text"):
                    text_input = gr.Textbox(
                        label="What did the person say?",
                        placeholder="Type the sentence here…",
                        lines=5,
                    )
                    gr.HTML('<p style="font-family:\'DM Mono\',monospace;font-size:0.62rem;letter-spacing:0.18em;text-transform:uppercase;color:#71717a;margin:14px 0 8px;">Quick examples</p>')
                    with gr.Row():
                        for ex in EXAMPLES:
                            gr.Button(
                                ex["label"],
                                elem_classes=["example-pill"],
                                size="sm",
                            ).click(
                                fn=lambda t=ex["text"]: t,
                                inputs=[],
                                outputs=[text_input],
                                queue=False,
                            )

                with gr.Tab("🎙️  Upload Audio"):
                    audio_input = gr.Audio(
                        type="filepath",
                        label="Upload a short audio clip (wav / mp3 / m4a)",
                    )
                    gr.HTML("""
                    <div class="audio-callout">
                        🎙️ <strong>Whisper AI</strong> auto-transcribes your clip.
                        The transcript feeds directly into the same sentiment + fusion pipeline,
                        giving you a full <strong>3-modality analysis</strong> — face, voice &amp; text.
                        Max 30 seconds recommended.
                    </div>
                    """)
                    transcript_out = gr.Textbox(
                        label="Auto-transcript (Whisper output)",
                        interactive=False,
                        lines=3,
                        placeholder="Transcript appears here after analysis…",
                    )

            analyse_btn = gr.Button("⚡  Run Analysis", variant="primary", size="lg")

    gr.HTML('<div class="msai-divider" style="margin-top:32px;"></div>')

    # ── Emotion & Sentiment ───────────────────────────────────────────────────
    gr.HTML('<p class="msai-section-label">02 &nbsp;/&nbsp; Emotion &amp; Sentiment</p>')

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 👁️ Visual Emotion")
            emotion_chart_out = gr.Image(label="Emotion Confidence Chart", height=260)
            emotion_label_out = gr.Textbox(label="Top Emotion", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("#### 💬 Textual Sentiment")
            sentiment_chart_out = gr.Image(label="Sentiment Confidence Chart", height=260)
            sentiment_label_out = gr.Textbox(label="Top Sentiment", interactive=False)

    gr.HTML('<div class="msai-divider" style="margin-top:32px;"></div>')

    # ── Fusion & Summary ──────────────────────────────────────────────────────
    gr.HTML('<p class="msai-section-label">03 &nbsp;/&nbsp; Fusion &amp; Summary</p>')

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["badge-glow"]):
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

    gr.HTML('<div class="msai-divider" style="margin-top:32px;"></div>')

    # ── Attention Map ─────────────────────────────────────────────────────────
    gr.HTML('<p class="msai-section-label">04 &nbsp;/&nbsp; Attention Map</p>')

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 🔥 Grad-CAM Heatmap")
            gradcam_out = gr.Image(
                label="Face regions driving the emotion prediction",
                height=300,
            )

    gr.HTML('<div class="msai-divider" style="margin-top:32px;"></div>')

    # ── History ───────────────────────────────────────────────────────────────
    gr.HTML('<p class="msai-section-label">05 &nbsp;/&nbsp; Session History</p>')
    history_out = gr.Markdown(
        value=_render_history(),
        elem_classes=["history-wrap"],
    )

    # ── Wire ──────────────────────────────────────────────────────────────────
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
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
from utils.helpers import make_bar_chart, format_confidence
from utils.gradcam import generate_gradcam
from config import APP_TITLE, APP_DESCRIPTION


def analyse(image, text):
    """Full multimodal analysis pipeline."""

    if image is None:
        msg = "Please upload a face photo."
        return None, msg, None, msg, msg, msg, None

    if not text or not text.strip():
        msg = "Please enter some text."
        return None, msg, None, msg, msg, msg, None

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

    # Step 4: Summary
    summary = generate_summary(fusion_result)

    # Step 5: Grad-CAM
    gradcam_image = generate_gradcam(image, emotion_result["top_emotion"])

    return (
        emotion_chart,
        emotion_label,
        sentiment_chart,
        sentiment_label,
        badge,
        summary,
        gradcam_image,
    )


with gr.Blocks(title="MoodSyncAI", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# MoodSyncAI")
    gr.Markdown(
        "**Multi-Modal Sentiment & Emotion Analyser**  \n"
        "Upload a face photo and type what the person said. "
        "The system analyses facial emotion, text sentiment, and detects any mismatch between them."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Face Photo",
                height=300,
            )
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="What did the person say?",
                placeholder="Type the sentence here...",
                lines=4,
            )
            analyse_btn = gr.Button(
                "Analyse",
                variant="primary",
                size="lg",
            )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Visual Emotion")
            emotion_chart_out = gr.Image(
                label="Emotion Confidence Chart",
                height=280,
            )
            emotion_label_out = gr.Textbox(
                label="Top Emotion",
                interactive=False,
            )

        with gr.Column(scale=1):
            gr.Markdown("### Textual Sentiment")
            sentiment_chart_out = gr.Image(
                label="Sentiment Confidence Chart",
                height=280,
            )
            sentiment_label_out = gr.Textbox(
                label="Top Sentiment",
                interactive=False,
            )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Fusion Result")
            badge_out = gr.Textbox(
                label="Mismatch Detection",
                interactive=False,
            )

        with gr.Column(scale=2):
            gr.Markdown("### Generative Summary")
            summary_out = gr.Textbox(
                label="Emotional Assessment",
                lines=4,
                interactive=False,
            )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔥 Grad-CAM Attention Map")
            gradcam_out = gr.Image(
                label="Which face regions influenced the emotion prediction",
                height=300,
            )

    analyse_btn.click(
        fn=analyse,
        inputs=[image_input, text_input],
        outputs=[
            emotion_chart_out,
            emotion_label_out,
            sentiment_chart_out,
            sentiment_label_out,
            badge_out,
            summary_out,
            gradcam_out,
        ]
    )


if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=True,
    )
---
title: MoodSyncAI
emoji: 🎭
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
---

# 🎭 MoodSyncAI — Multimodal Sentiment & Emotion Analysis

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/supreethadevagiri/moodsyncai)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.29-orange)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A multimodal AI system that analyses emotion and sentiment from **face images**, **text**, and **audio** simultaneously — detecting mismatches between what someone says and how they look or sound.

---

## 🚀 Live Demo

👉 **[Try it on HuggingFace Spaces](https://huggingface.co/spaces/supreethadevagiri/moodsyncai)**

---

## 🧠 How It Works

MoodSyncAI takes up to three inputs and runs them through specialised AI models:

```
📸 Face Image / 📷 Webcam  →  CNN (DeepFace/FER)       →  Visual Emotion
💬 Text Input  →  RoBERTa Transformer      →  Textual Sentiment
🎙️ Audio Clip  →  Whisper → RoBERTa        →  Audio Sentiment
                        ↓
              🔀 Multimodal Fusion Layer
                        ↓
         ✅ ALIGNED  /  ⚠️ MISMATCH DETECTED
                        ↓
         📝 GPT-2 Generative Summary
```

### Example

| Input | Result |
|-------|--------|
| Face photo (stressed expression) | Visual Emotion: **Sad / Fearful — 68%** |
| Text: *"No, I think the project is going really well."* | Textual Sentiment: **Positive — 81%** |
| Fusion result | ⚠️ **MISMATCH DETECTED** |
| Generative summary | *"Despite expressing positive sentiment verbally, the speaker's facial cues indicate stress or discomfort."* |

---

## 🏗️ Architecture

### Models Used

| Modality | Model | Purpose |
|----------|-------|---------|
| 👁️ Vision | DeepFace + FER (ResNet backbone) | Facial emotion classification |
| 💬 Text | RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment`) | Sentiment analysis |
| 🎙️ Audio | OpenAI Whisper | Speech-to-text transcription |
| 🔀 Fusion | Neural Fusion Layer (trained MLP) | Multimodal alignment/mismatch detection |
| 📝 Generation | GPT-2 | Natural language emotional summary |
| 🔍 Explainability | Grad-CAM | Visual attention heatmap on face |

### Fusion Strategy

- **Neural Fusion** — a small trained MLP combines the visual and textual embedding vectors to produce a unified prediction
- **Mismatch Detection** — if visual emotion polarity and textual sentiment polarity diverge beyond a threshold, an amber MISMATCH badge is triggered
- **Three-modality mode** — when audio is uploaded, Whisper transcribes it and the transcript is fed into the same text sentiment pipeline, then all three signals are fused

---

## ✨ Features

- 📸 **Image upload** — analyse facial emotion from any photo
- 📷 **Webcam capture** — take a live snapshot directly from your camera for instant analysis
- 💬 **Text input** — type or paste what the person said
- 🎙️ **Audio upload** — upload a `.wav` / `.mp3` clip; Whisper auto-transcribes it
- 🔀 **Multimodal fusion** — combines all signals with mismatch detection
- 📊 **Confidence charts** — bar charts across all emotion/sentiment categories
- 🔍 **Grad-CAM heatmap** — highlights which facial regions influenced the prediction
- 📝 **Generative summary** — GPT-2 produces a plain-language emotional context
- 🕓 **Session history** — keeps last 3 analyses in a results table
- 💡 **Example inputs** — one-click preset scenarios for quick demos

---

## 🛠️ Tech Stack

- **Frontend / UI** — Gradio 5.29
- **Vision** — DeepFace, FER, TensorFlow 2.20, OpenCV
- **NLP** — HuggingFace Transformers, RoBERTa, GPT-2
- **Audio** — OpenAI Whisper, Librosa
- **Explainability** — grad-cam (Grad-CAM for CNN)
- **Deep Learning** — PyTorch 2.6, TorchVision
- **Deployment** — HuggingFace Spaces

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/supreethadevagiri/MoodSyncAI.git
cd MoodSyncAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

> **Note:** ffmpeg is required for non-WAV audio formats.
> Install with: `brew install ffmpeg` (Mac) or `sudo apt install ffmpeg` (Linux)

---

## 📁 Project Structure

```
MoodSyncAI/
├── app.py                          # Main Gradio application
├── requirements.txt                # Python dependencies
├── models/
│   ├── emotion_detector.py         # CNN facial emotion (DeepFace/FER)
│   ├── sentiment_analyzer.py       # RoBERTa text sentiment
│   ├── audio_transcriber.py        # Whisper audio transcription
│   ├── fusion.py                   # Multimodal fusion layer
│   ├── generator.py                # GPT-2 generative summary
│   └── gradcam.py                  # Grad-CAM explainability
└── README.md
```

---

## ✨ Highlights

- 🔍 **Grad-CAM explainability** — visualises which facial regions drive the emotion prediction
- 🧠 **Learned neural fusion** — a trained MLP combines visual + text embeddings for smarter alignment detection
- 🎙️ **Three-modality support** — face, text, and audio all processed in a single pipeline via Whisper
- 📷 **Live webcam capture** — real-time snapshot analysis without needing to upload a photo
- 🚀 **Deployed & publicly accessible** on HuggingFace Spaces

---

## 👩‍💻 Author

**Supreetha Devagiri**

---

## 📄 License

This project is licensed under the MIT License.

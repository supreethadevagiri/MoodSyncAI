# ─────────────────────────────────────────────
# MoodSyncAI – config.py
# Central configuration file.
# Change settings here instead of hunting through all files.
# ─────────────────────────────────────────────

# ── Text Sentiment Model ──────────────────────────────────────────────────────
# We use a pretrained RoBERTa model fine-tuned on Twitter sentiment data.
# It classifies text into: NEGATIVE / NEUTRAL / POSITIVE
TEXT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# ── Face Emotion Model ────────────────────────────────────────────────────────
# DeepFace supports multiple backends. 'opencv' is fastest for demos.
# Options: "opencv", "ssd", "dlib", "mtcnn", "retinaface"
FACE_DETECTOR_BACKEND = "opencv"

# The emotion model inside DeepFace (no change needed)
# It detects: angry, disgust, fear, happy, sad, surprise, neutral
FACE_MODEL_NAME = "Emotion"

# ── Fusion Settings ───────────────────────────────────────────────────────────
# How much weight to give each modality when combining scores (must add to 1.0)
VISUAL_WEIGHT = 0.5       # 50% weight to face emotion
TEXT_WEIGHT   = 0.5       # 50% weight to text sentiment

# If the top emotion from face and text disagree by this much → flag MISMATCH
MISMATCH_THRESHOLD = 0.3  # 30% difference triggers mismatch warning

# ── Generative Summary ────────────────────────────────────────────────────────
# We use GPT-2 (runs locally, no API key needed) to generate the summary.
# If you have an OpenAI/Anthropic API key, you can swap this out later.
GENERATOR_MODEL = "gpt2"
GENERATOR_MAX_NEW_TOKENS = 120
GENERATOR_TEMPERATURE = 0.7   # Higher = more creative, Lower = more focused

# ── Emotion & Sentiment Label Maps ───────────────────────────────────────────
# Map DeepFace emotion labels to a simpler positive/neutral/negative group
EMOTION_VALENCE_MAP = {
    "happy":    "positive",
    "surprise": "positive",
    "neutral":  "neutral",
    "sad":      "negative",
    "angry":    "negative",
    "disgust":  "negative",
    "fear":     "negative",
}

# Map RoBERTa output labels to clean display labels
SENTIMENT_LABEL_MAP = {
    "positive": "Positive 😊",
    "neutral":  "Neutral 😐",
    "negative": "Negative 😔",
}

# Map DeepFace emotion labels to display labels with emoji
EMOTION_LABEL_MAP = {
    "happy":    "Happy 😄",
    "sad":      "Sad 😢",
    "angry":    "Angry 😠",
    "fear":     "Fearful 😨",
    "surprise": "Surprised 😲",
    "disgust":  "Disgusted 🤢",
    "neutral":  "Neutral 😐",
}

# ── UI Settings ───────────────────────────────────────────────────────────────
APP_TITLE       = "MoodSyncAI 🧠"
APP_DESCRIPTION = (
    "Multi-Modal Sentiment & Emotion Analyser\n"
    "Upload a face photo + type what the person said → get a full emotional analysis."
)
APP_THEME       = "soft"   # Gradio theme options: soft, default, glass, monochrome

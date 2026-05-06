# ─────────────────────────────────────────────
# MoodSyncAI – utils/helpers.py
# Small reusable helper functions used across the project.
# ─────────────────────────────────────────────

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image → NumPy array (what DeepFace and OpenCV expect).
    
    PIL Image is what Gradio gives us when a user uploads a photo.
    NumPy array is what our models need.
    """
    return np.array(image)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """
    Convert a NumPy array → PIL Image (what Gradio displays).
    """
    return Image.fromarray(array.astype(np.uint8))


def softmax(scores: dict) -> dict:
    """
    Turn raw model scores into probabilities that add up to 100%.
    
    Example:
        Input:  {"happy": 2.1, "sad": 0.3, "angry": -0.5}
        Output: {"happy": 0.72, "sad": 0.20, "angry": 0.08}
    """
    values = np.array(list(scores.values()), dtype=np.float32)
    exp_values = np.exp(values - np.max(values))   # subtract max for numerical stability
    probabilities = exp_values / exp_values.sum()
    return {k: float(round(v, 4)) for k, v in zip(scores.keys(), probabilities)}


def make_bar_chart(label_scores: dict, title: str, color: str = "#4F86C6") -> Image.Image:
    """
    Create a horizontal bar chart of emotion/sentiment scores.
    Returns a PIL Image that Gradio can display directly.

    Args:
        label_scores : dict like {"happy": 0.68, "sad": 0.12, ...}
        title        : chart title string
        color        : bar color hex code
    """
    labels = list(label_scores.keys())
    values = [v * 100 for v in label_scores.values()]   # convert to percentages

    fig, ax = plt.subplots(figsize=(6, max(3, len(labels) * 0.6)))
    bars = ax.barh(labels, values, color=color, edgecolor="white", height=0.55)

    # Add percentage labels inside bars
    for bar, val in zip(bars, values):
        ax.text(
            min(val + 1, 95), bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9, color="black"
        )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()

    # Save figure to a bytes buffer, then open as PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def format_confidence(value: float) -> str:
    """
    Format a float (0.0 – 1.0) as a readable percentage string.
    Example: 0.683 → "68.3%"
    """
    return f"{value * 100:.1f}%"


def get_top_label(scores: dict) -> tuple[str, float]:
    """
    Return the label with the highest score and its value.
    Example: {"happy": 0.68, "sad": 0.20} → ("happy", 0.68)
    """
    top = max(scores, key=scores.get)
    return top, scores[top]

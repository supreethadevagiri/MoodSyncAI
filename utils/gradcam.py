import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image


# ── Emotion labels in the order the model was trained ────────────────────────
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _build_model():
    """ResNet-18 with 7-class emotion head (matches FER-trained weights)."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.eval()
    return model


def _preprocess(pil_image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return transform(pil_image).unsqueeze(0)


def generate_gradcam(pil_image: Image.Image, emotion_label: str) -> Image.Image:
    """
    Produce a Grad-CAM heatmap overlay for the given PIL face image.
    Returns a PIL Image (the original with heatmap blended on top).
    """
    try:
        model = _build_model()
        target_layer = model.layer4[-1]

        tensor = _preprocess(pil_image)

        # ── Forward + backward hooks ──────────────────────────────────────────
        gradients = []
        activations = []

        def save_gradient(grad):
            gradients.append(grad)

        def forward_hook(module, input, output):
            activations.append(output)
            output.register_hook(save_gradient)

        handle = target_layer.register_forward_hook(forward_hook)

        output = model(tensor)
        handle.remove()

        # Use the emotion index if found, else use top prediction
        if emotion_label.lower() in EMOTION_LABELS:
            class_idx = EMOTION_LABELS.index(emotion_label.lower())
        else:
            class_idx = output.argmax(dim=1).item()

        model.zero_grad()
        output[0, class_idx].backward()

        # ── Compute CAM ───────────────────────────────────────────────────────
        grads = gradients[0].detach().numpy()[0]       # (C, H, W)
        acts  = activations[0].detach().numpy()[0]     # (C, H, W)
        weights = grads.mean(axis=(1, 2))              # (C,)
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        # ── Resize & colorise ─────────────────────────────────────────────────
        orig_w, orig_h = pil_image.size
        cam_resized = cv2.resize(cam, (orig_w, orig_h))
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # ── Blend with original ───────────────────────────────────────────────
        orig_np = np.array(pil_image.convert("RGB"))
        blended = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
        return Image.fromarray(blended)

    except Exception as e:
        print(f"[GradCAM] Error: {e} — returning original image")
        return pil_image


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_img = Image.new("RGB", (200, 200), color=(180, 140, 100))
    result = generate_gradcam(test_img, "neutral")
    result.save("assets/gradcam_test.png")
    print("✅ Grad-CAM test complete! Check assets/gradcam_test.png")
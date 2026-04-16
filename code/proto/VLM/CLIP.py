import open_clip
import torch
from PIL import Image
import os, sys
from pathlib import Path

# ── Load once at module level ─────────────────────────────────────────────────
os.environ["HUGGINGFACE_HUB_CACHE"] = "/Volumes/T7_AO/clip-models"
device = "mps" if torch.backends.mps.is_available() else "cpu"

"""
CLIP ViT-B-32   350 MB   RAM ~1 GB   ~50ms/image
CLIP ViT-B-16   350 MB   RAM ~1 GB   ~80ms/image
CLIP ViT-L-14   890 MB   RAM ~2 GB   ~150ms/image
"""

print("⏳ Processing model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai",
    cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE", "~/.cache"),
    quick_gelu=True
)
model.eval().to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
print(f"✅ CLIP loaded on {device}")


def ask_question_on_image(image_path: str, candidates: list[str]) -> dict:
    """
    Instead of a free-form question, provide a list of candidate descriptions.
    Returns a score (0-1) for each candidate.

    Example:
        ask_question_on_image("frame.png", [
            "a robot standing upright",
            "a robot fallen on the ground",
        ])
        → {"a robot standing upright": 0.82, "a robot fallen on the ground": 0.18}
    """
    if not Path(image_path).exists():
        print(f"❌ Image not found : {image_path}")
        sys.exit(1)

    # Preprocess image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Tokenize candidates
    texts = tokenizer(candidates).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(texts)

        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Similarity scores
        probs = (image_features @ text_features.T).softmax(dim=-1)[0]

    return {label: round(float(score), 3) for label, score in zip(candidates, probs)}

if __name__ == "__main__":

    diagnostic_candidates = [
        # ── Scene / render style ──────────────────────────────────────────
        "a 3D physics simulation screenshot with blue background",
        "a side view render of a simulated creature on a flat surface",
        "a game engine screenshot of a robot on a dark ground line",
        "a minimalist 3D animation of a stick figure robot",

        # ── Torso state ───────────────────────────────────────────────────
        "a white oval shape connected to colored sticks near the ground",
        "a robot torso parallel to the ground about to fall",
        "a robot body held upright above the ground",
        "a white capsule shape tilted at 45 degrees",

        # ── Posture / fall state ──────────────────────────────────────────
        "a robot in a low crouching position close to the ground",
        "a robot fallen with its body touching the ground line",
        "a robot standing tall with legs extended downward",
        "a robot mid-fall with torso nearly horizontal",

        # ── Limb configuration ────────────────────────────────────────────
        "colored sticks arranged like legs supporting a white body",
        "green and yellow rods connected to a central white shape",
        "multiple limbs splayed outward from a central body",
        "legs bent under a body close to the floor",
    ]

    answer = ask_question_on_image("./img/spider.png", diagnostic_candidates)
    for label, score in sorted(answer.items(), key=lambda x: -x[1]):
        print(f"{score:.3f}  {label}")
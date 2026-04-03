import ollama
import base64
import json
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path


def extract_frames(video_path: str, n_frames_per_sec: int = 4) -> list[str]:
    """Extrait n frames régulièrement espacées de la vidéo."""
    tmp_dir = tempfile.mkdtemp()
    output_pattern = f"{tmp_dir}/frame_%03d.jpg"

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={n_frames_per_sec}",  # Prend toutes les frames (8fps natif)
        "-q:v", "2",  # Qualité max, les frames sont déjà petites
        output_pattern,
        "-loglevel", "error"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Erreur ffmpeg : {result.stderr}")
        sys.exit(1)

    frames = sorted(Path(tmp_dir).glob("frame_*.jpg"))
    print(f"✅ {len(frames)} frames extraites")
    return [str(f) for f in frames], tmp_dir

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def score_robot_video(video_path: str, n_frames: int = 8) -> dict:
    if not Path(video_path).exists():
        print(f"❌ Vidéo non trouvée : {video_path}")
        sys.exit(1)

    print(f"🎬 Analyse de : {video_path}")

    # Extraction des frames
    frame_paths, tmp_dir = extract_frames(video_path, n_frames)

    try:
        # Encodage de toutes les frames
        images_b64 = [encode_image(f) for f in frame_paths]
        print(f"⏳ Envoi de {len(images_b64)} frames au modèle...")

        prompt = f"""Tu regardes {len(images_b64)} frames extraites d'une simulation de 5 secondes montrant un robot sur sol bleu en mouvement. Son objectif est de se déplacé en restant debout sur ses jambes de couleurs différentes. Les frames sont ordonnées chronologiquement.

Etablis en commentaire décrivant le comportement factuel observé sans faire  de supposition.
Le robot est considéré comme tombé si son torse (en gris) touche le sol. Si le robot ne bouge pas, c'est mauvais.

Évalue cette morphologie et ce mouvement sur les critères suivants :
- Dynamisme du mouvement (non statique) (0-10)
- Stabilité apparente (ne tombe pas) (0-10)
- Efficacité locomotrice estimée (vitesse et simplicité) (0-10)
- Score global d'intérêt (pottentiel evolutif) (0-10)

Réponds UNIQUEMENT avec un JSON valide, sans texte avant ou après, exemple :
{{"fluidite": 7, "stabilite": 8, "efficacite": 6, "interet": 7, "commentaire": "..."}}"""

        response = ollama.chat(
            model="qwen2.5vl:7b",
            messages=[{
                "role": "user",
                "content": prompt,
                "images": images_b64      # Toutes les frames en une requête
            }]
        )

        text = response["message"]["content"]

        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            print("❌ Pas de JSON trouvé dans la réponse")
            print(f"🤖 Réponse brute : {text}\n")
            sys.exit(1)

        return json.loads(text[start:end])

    finally:
        # Nettoyage des frames temporaires
        shutil.rmtree(tmp_dir)
        print("🧹 Frames temporaires supprimées")

def ask_question_on_video(video_path: str, question: str, n_frames: int = 8) -> dict:
    if not Path(video_path).exists():
        print(f"❌ Vidéo non trouvée : {video_path}")
        sys.exit(1)

    print(f"🎬 Analyse de : {video_path}")

    # Extraction des frames
    frame_paths, tmp_dir = extract_frames(video_path, n_frames)

    try:
        # Encodage de toutes les frames
        images_b64 = [encode_image(f) for f in frame_paths]
        print(f"⏳ Envoi de {len(images_b64)} frames au modèle...")

        prompt = f"""
        You are looking at {len(images_b64)} frames taken from a 5 second simulation showing a robot moving on a blue floor. Its goal is to move while remaining upright on its legs, which are of different colors and attached to its gray torso. The frames are arranged in chronological order.
        Focus solely on this question and answer it briefly: 
        {question}"""

        response = ollama.chat(
            model="qwen2.5vl:7b",
            messages=[{
                "role": "user",
                "content": prompt,
                "images": images_b64      # Toutes les frames en une requête
            }]
        )

        text = response["message"]["content"]

        shutil.rmtree(tmp_dir)
        print("🧹 Frames temporaires supprimées")

        return text
    finally:
        # Nettoyage des frames temporaires
        #shutil.rmtree(tmp_dir)
        #print("🧹 Frames temporaires supprimées")
        pass

def ask_question_on_image(image_path: str, question: str) -> dict:
    if not Path(image_path).exists():
        print(f"❌ Image non trouvée : {image_path}")
        sys.exit(1)

    try:
        # Encodage de toutes les frames
        images_b64 = [encode_image(image_path)]
        print(f"⏳ Envoi de {len(images_b64)} frames au modèle...")

        prompt = f"""
        You are looking at {len(images_b64)} frames taken from a simulation showing a robot.
        Focus solely on this question and answer it briefly: 
        {question}"""

        response = ollama.chat(
            model="qwen2.5vl:7b",
            messages=[{
                "role": "user",
                "content": prompt,
                "images": images_b64      # Toutes les frames en une requête
            }]
        )

        text = response["message"]["content"]
        return text
    finally:
        pass

def model_test(model) -> dict:
    try:
        prompt = f"""
        This is a test prompt, answer "All good" if everything works.
        """

        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
            }]
        )

        text = response["message"]["content"]

        print(f"Test of {model} : {text}")
    finally:
        pass

if __name__ == "__main__":
    model_test("qwen2.5vl:7b")

    #resp = ask_question_on_image("robot.png", "What do you see ?")
    #print(f"Question : {"What do you see ?"} \nAnswer : {resp}")

    video_path = sys.argv[1] if len(sys.argv) > 1 else "fall_1.mp4"
    n_frames_per_sec = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    question = "Did the robot fall ? (fallen = not moving and torso touching the ground)"
    resp = ask_question_on_video(video_path, question, n_frames_per_sec)
    print(f"Question : {question} \nAnswer : {resp}")

    # result = score_robot_video(video_path, n_frames_per_sec)
    #
    # print("✅ Scores extraits :")
    # print(f"  Fluidité      : {result.get('fluidite', '?')}/10")
    # print(f"  Stabilité     : {result.get('stabilite', '?')}/10")
    # print(f"  Efficacité    : {result.get('efficacite', '?')}/10")
    # print(f"  Intérêt       : {result.get('interet', '?')}/10")
    # print(f"  Commentaire   : {result.get('commentaire', '')}")
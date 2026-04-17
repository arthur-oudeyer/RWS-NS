from google import genai
import PIL.Image

# ── Config ────────────────────────────────────────────────────────────────────
from code.api_keys import APIKEY_GEMINI
API_KEY = APIKEY_GEMINI
MODEL   = "gemini-3.1-flash-lite-preview"

client = genai.Client(api_key=API_KEY)

# Charge ton image
image = PIL.Image.open("./img/morph_0008.png")

# Ton prompt de scoring
TARGET_INSPIRATION = "insect"
FORMAT = """
    {
      "observation":  "factual decription",
      "interpretation":  "interpretation description and explanation",
      "coherence":   { "score": X, "reason": "..." },
      "originality": { "score": X, "reason": "..." },
      "interest":    { "score": X, "reason": "..." }
    }
    """

prompt = f"""
    ═══ CONTEXT ═══

    You are a strict and skeptical evaluator analyzing a static image of a MuJoCo robot morphology.
    Your job is to be PRECISE and reproduce human-like feedback on the robot's structural design.

    The scene:
    - 2 simultaneous views of the same morphology: left = front/side angle, right = 3/4 perspective
    - dark/grey checkerboard floor
    - Robot has a white cylindrical torso and colored limbs (red, yellow, green, purple...)
    - The robot's locomotion objective: move forward continuously while staying upright
    - The robot's morphology objective: looking like an {TARGET_INSPIRATION} (= target)

    ═══ ANALYSIS ═══

    Step 1 — Factual observation
    Describe precisely what you see in both views:
    - Torso shape, size and position relative to the ground
    - Number of limbs, their attachment points, segment lengths and approximate angles
    - Overall stance: is the robot upright, crouching, sprawled, collapsed?
    - Any asymmetry or unusual structural feature across the two views (shapes, connections, ..)

    Step 2 — Morphology interpretation
    You are evaluating structural design.
    Based on the static pose and limb layout:

    - Does the morphology resemble {TARGET_INSPIRATION}? Identify which features do or do not match.
    - (e.g. for elephant: is there a trunk-like limb? Are legs thick and pillar-like?)
    - Does the structure suggest stable locomotion is even physically plausible?
      Consider: center of mass, ground contact points, limb symmetry, joint range of motion (~90°).
    - If the morphology shows originality or promising structural traits, state what they are
      and how they could support efficient locomotion.
    - If the morphology is poorly designed, state specifically why
      (e.g. too few contact points, limbs too short to reach ground, torso too high).

    Step 3 — Score
    Score each dimension using only the static image evidence.
    Be conservative. Do not infer runtime behavior from a single frame.

    SCORING RULES:

    coherence  — How well does the morphology match a {TARGET_INSPIRATION}?
      0–2  = no recognizable similarity to a {TARGET_INSPIRATION}
      3–4  = vague resemblance, one weak matching feature
      5–6  = partial match, 1–2 clear {TARGET_INSPIRATION}-like features present
      7–8  = strong resemblance, most key features identifiable
      9–10 = unmistakable likeness, structurally faithful to a {TARGET_INSPIRATION}

    originality  — Is the structural design novel or inventive?
      0–2  = generic, indistinguishable from a randomly generated MuJoCo morphology
      3–4  = basic organisation and minor variation on a standard body plan
      5–6  = one interesting structural choice (unusual limb count, asymmetry, etc.)
      7–8  = clearly novel design with multiple inventive features
      9–10 = highly creative, unexpected combination of structures

    interest  — Evolutionary/locomotion potential from structural analysis alone
      0–2  = structurally implausible: cannot stand, no viable contact points
      3–4  = poor design but not hopeless; major locomotion issues likely
      5–6  = plausible but inefficient; gait would be limited or unstable
      7–8  = solid design; structure suggests stable and potentially efficient gait
      9–10 = excellent design; high locomotion potential, well-suited to target morphology

    ═══ OUTPUT FORMAT ═══
    {FORMAT}
    """

# Compte les tokens AVANT d'envoyer la vraie requête (gratuit)
token_count = client.models.count_tokens(
    model="gemini-3.1-flash-lite-preview",
    contents=[prompt, image]
)
print(f"---> Tokens en entrée : {token_count.total_tokens}")

# Optionnel : envoie la vraie requête et vérifie l'usage réel
response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents=[prompt, image]
)
print(f"---> Usage réel — input: {response.usage_metadata.prompt_token_count}, "
      f"output: {response.usage_metadata.candidates_token_count}")
print(f"Réponse : {response.text}")

try:
    print("✅ Scores :")
    print(f"  Observation     : {response.text.get('observation')}")
    print(f"  Interpretation  : {response.text.get('interpretation')}")
    print(f"  Coherence       : {response.text.get('coherence')}")
    print(f"  Originality     : {response.text.get('originality')}")
    print(f"  Interest        : {response.text.get('interest')}")
except:
    pass
from google import genai
from google.genai import types
import json
import sys
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
from code.api_keys import APIKEY_GEMINI
API_KEY = APIKEY_GEMINI

"""
Gemini 3.1 Flash-Lite -> gemini-3.1-flash-lite-preview
Gemini 3 Flash        -> gemini-3-flash-preview
Gemini 3.1 Pro        -> gemini-3.1-pro-preview
"""
MODEL   = "gemini-3.1-pro-preview"

client = genai.Client(api_key=API_KEY)

# ── Upload + scoring ───────────────────────────────────────────────────────────
def score_robot_video(video_path: str) -> dict:
    if not Path(video_path).exists():
        print(f"❌ Video not found : {video_path}")
        sys.exit(1)

    print(f"🎬 Uploading : {video_path}")
    video_file = client.files.upload(
        file=video_path,
        config=types.UploadFileConfig(mime_type="video/mp4")
    )

    # Wait for processing
    print("⏳ Waiting for video processing...")
    while video_file.state.name == "PROCESSING":
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        print("❌ Video processing failed")
        sys.exit(1)

    print("✅ Video ready, sending to model...")

    prompt = """You are a strict and skeptical evaluator analyzing a 5-second physics simulation video of a robot. Your job is to be PRECISE and CONSERVATIVE — do not give benefit of the doubt, do not assume movement if unsure.

    The scene:
    - Fixed camera, no camera movement
    - Blue/dark grey checkerboard floor (use floor tiles as a position reference grid) / may be seen as a dark line with the camera angle.
    - Robot has a grey/white cylindrical torso and colored legs (red, yellow, green...)
    - The robot's goal: move forward continuously while staying upright

    ═══ ANALYSIS RULES ═══

    FALL DETECTION (highest priority):
    - FALLEN = torso cylinder touching or nearly touching the floor
    - FALLEN = colored segments all lying roughly horizontal/flat
    - If the torso is clearly tilted >45° from vertical → mark as tilted, likely fallen
    - When in doubt → call it fallen. False negative (missing a fall) is worse than false positive.

    MOVEMENT DETECTION (be strict):
    - Use floor tiles as reference. Count how many tiles the robot has crossed.
    - A robot that oscillates in place (legs moving but body not translating) = NOT moving efficiently
    - A robot that made 1-2 steps then stopped = "moved briefly then stopped"
    - CRITICAL: Compare the robot's position on the floor grid between first frame and last frame.
      If the position is the same or nearly the same → the robot did NOT move forward effectively.
    - Do NOT describe movement as "continuous" unless the robot clearly advances across multiple tiles throughout the full 5 seconds.

    STAGNATION DETECTION:
    - Split the video mentally into two halves: first 2.5s and last 2.5s
    - If the robot moved in the first half but is static or stuck in the second half → explicitly note "stagnated at ~Xs"
    - A stuck robot holding a stable pose is NOT locomotion, it is stagnation.

    SCORING RULES (be conservative, do not overestimate):
    - dynamism: 
        0-2 = never moved or moved less than 1 tile total
        3-4 = moved briefly (1-2 steps) then stopped/stagnated
        5-6 = moved intermittently, not consistently
        7-8 = moved consistently for most of the 5 seconds
        9-10 = continuous fluid movement across the full video
    - stability:
        0 = fallen before 1s
        1-2 = fallen between 1-2s
        3-4 = fallen between 2-4s
        5-6 = did not fall but severely tilted / barely upright
        7-8 = stayed upright with some wobble
        9-10 = perfectly stable throughout
    - efficiency:
        0-2 = no forward displacement or fell immediately
        3-4 = minor displacement, inefficient gait
        5-6 = moderate displacement with visible effort
        7-8 = clear forward progress with reasonable gait
        9-10 = fast, smooth, energy-efficient locomotion
    - interest (evolutionary potential):
        Consider: did it show ANY promising behavior even briefly?
        A robot that made one good step before stagnating = moderate interest (4-5)
        A robot that moved well but fell = moderate interest (5-6)
        A robot that never moved = low interest (1-3)
        A robot that moved consistently = high interest (7-9)

    ═══ OUTPUT FORMAT ═══

    Step 1 — Frame-by-frame factual observation:
    Be specific. Reference floor tiles for position. Note exact moments of events.
    - Frame 1 (0.0s): [posture, position on grid]
    - Mid video (~2.5s): [posture, position, still moving?]
    - Final frame (5.0s): [posture, final position, same tile as start?, same tile as mid video?]
    - Key events: [fall at Xs / stagnation at Xs / direction change at Xs]

    Step 2 — Conservative scores.

    Respond ONLY with valid JSON, no text before or after:
    {
      "fallen": false,
      "fall_moment_s": null,
      "stagnation_moment_s": null,
      "tiles_crossed": 0,
      "dynamism": 4,
      "stability": 6,
      "efficiency": 3,
      "interest": 5,
      "comment": "precise factual description referencing tiles and timestamps"
    }"""

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_uri(
                file_uri=video_file.uri,
                mime_type="video/mp4"
            ),
            prompt
        ]
    )

    text = response.text

    # Cleanup
    client.files.delete(name=video_file.name)
    print("🧹 Remote file deleted")

    # Extract JSON
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        print("❌ No JSON found in response")
        print(f"🤖 Raw response :\n{text}\n")
        sys.exit(1)

    return json.loads(text[start:end])

def ask_question_on_image(image_path: str, question: str) -> dict:
    if not Path(image_path).exists():
        print(f"❌ Image not found : {image_path}")
        sys.exit(1)

    print(f"🎬 Uploading : {image_path}")
    image_file = client.files.upload(
        file=image_path,
        config=types.UploadFileConfig(mime_type="image/png")
    )

    # Wait for processing
    print("⏳ Waiting for image processing...")
    while image_file.state.name == "PROCESSING":
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)

    if image_file.state.name == "FAILED":
        print("❌ image processing failed")
        sys.exit(1)

    print("✅ Image ready, sending to model...")

    try:
        prompt = f"""
                You are looking at frame taken from a 5 second simulation showing a simulated robot composed of a white torso and colored legs. It stands on a black floor, and the background is blue.
                Focus solely on this question and answer it briefly: 
                {question}"""

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_uri(
                    file_uri=image_file.uri,
                    mime_type="image/png"
                ),
                prompt
            ]
        )

        text = response.text

        return text
    finally:
        pass

def score_robot_image(image_path: str) -> dict:
    if not Path(image_path).exists():
        print(f"❌ Image not found : {image_path}")
        sys.exit(1)

    print(f"🎬 Uploading : {image_path}")
    img_file = client.files.upload(
        file=image_path,
        config=types.UploadFileConfig(mime_type="image/png")
    )

    # Wait for processing
    print("⏳ Waiting for image processing...")
    while img_file.state.name == "PROCESSING":
        time.sleep(1)
        img_file = client.files.get(name=img_file.name)

    if img_file.state.name == "FAILED":
        print("❌ Image processing failed")
        sys.exit(1)

    print("✅ Image ready, sending to model...")

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

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_uri(
                file_uri=img_file.uri,
                mime_type="image/png"
            ),
            prompt
        ]
    )

    text = response.text

    # Cleanup
    client.files.delete(name=img_file.name)
    print("🧹 Remote file deleted")

    # Extract JSON
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        print("❌ No JSON found in response")
        print(f"🤖 Raw response :\n{text}\n")
        sys.exit(1)

    return json.loads(text[start:end])


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    #question = "How does this robot morphology makes you think about ?"
    #resp = ask_question_on_image("./img/morph_0003.png", question)
    #print(f"Question : {question} \nAnswer : {resp}")

    resp = score_robot_image("./img/morph_0008.png")
    print("✅ Scores :")
    print(f"  Observation     : {resp.get('observation')}")
    print(f"  Interpretation  : {resp.get('interpretation')}")
    print(f"  Coherence       : {resp.get('coherence')}")
    print(f"  Originality     : {resp.get('originality')}")
    print(f"  Interest        : {resp.get('interest')}")

    # video_path = sys.argv[1] if len(sys.argv) > 1 else "./video/mid.mp4"
    # result = score_robot_video(video_path)
    #
    # print("✅ Scores :")
    # print(f"  Fallen        : {result.get('fallen')}")
    # print(f"  Fall moment   : {result.get('fall_moment')}")
    # print(f"  Dynamism      : {result.get('dynamism')}/10")
    # print(f"  Stability     : {result.get('stability')}/10")
    # print(f"  Efficiency    : {result.get('efficiency')}/10")
    # print(f"  Interest      : {result.get('interest')}/10")
    # print(f"  Comment       : {result.get('comment')}")
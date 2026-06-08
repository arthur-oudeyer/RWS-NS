from google import genai
from google.genai import types
import json
import sys, os
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from api_keys import APIKEY_GEMINI

API_KEY = APIKEY_GEMINI

"""
Gemini 3.1 Flash-Lite -> gemini-3.1-flash-lite-preview
Gemini 3 Flash        -> gemini-3-flash-preview
Gemini 3.1 Pro        -> gemini-3.1-pro-preview
"""
MODEL   = "gemini-3-flash-preview"

client = genai.Client(api_key=API_KEY)

# ── Upload + scoring ───────────────────────────────────────────────────────────
def score_robot_video(video_path: str, target_behaviour: str) -> dict:
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

    output_format = """\
        {
          "observation":    "key-time steps factual description (timestamps, floor-tile reference, posture of each limb, ...)",
          "interpretation": "behavioural interpretation relative to the target",
          "coherence":      { "score": <int 0-100>, "reason": "..." },
          "originality":    { "score": <int 0-100>, "reason": "..." },
          "potential":       { "score": <int 0-100>, "reason": "..." }
        }"""

    prompt = f"""
        ═══ CONTEXT ═══

        You are looking at video of a 5 second simulation showing two side-by-side view of a simulated robot composed of a white torso and colored legs. It stands on a green checkered floor, and the background is blue.

        Target behavior : {target_behaviour}

        ═══ ANALYSIS ═══

        Step 1 — factual observation
        Describe the robot morphology and behavior.

        Step 2 — Behavioural interpretation
        - Did the robot make consistent consistent action relevant with the target behavior ?
        - Was the gait coherent (periodic, balanced, repeatable) or random ? What was the type of the gait (smooth, energetic, nervous, wide, brutal, efficient, small, homogenous, ...) ?
        - Is there anything novel or interesting about the motion pattern even if the robot did not perform well for the target behavior ? (ex: is a limb doing a movement with great potential ?)

        Step 3 — scoring (each dimension 0–100)

        coherence — Is the gait relevant for the target behavior ?
          0–29   = chaotic thrashing, immediate collapse, fully static or no recognisable pattern
          30–49  = unstable, sporadic; one or two coherent moments only that have a link to the target
          50–69  = partial coherence; clear periodic pattern or specific movement but with wobble or stalls. The target can be identified.
          70–89  = coherent, repeatable gait or target well reached ; minor instabilities only. The intention toward target is obvious.
          90–100 = clean, stable, periodic locomotion throughout, the target is perfectly depict through this video.

        originality — Did the robot achieve something toward the behavioral target in an original way ?
          0–29   = no movement or movement very basic with no progress toward the target
          30–49  = one basic movement, not very original
          50–69  = novel movements that provide new ability for the robot
          70–89  = clear and unexpected movement that somehow help the robot progress toward the target behavior
          90–100 = very unexpected but very efficient way to reach the behavior wanted

        potential — Is the gait pattern interesting, biologically plausible and leads to a real evolutionary potential ?
          0–29   = uninteresting (random, fallen) or obviously broken
          30–49  = generic, predictable motion with no notable features
          50–69  = one notable element (unusual gait phase, rhythm, recovery) that have potential
          70–89  = clearly interesting motion: reminiscent of an animal gait,
                   coordinated pattern, or creative body usage to reached the target. There is a great potential.
          90–100 = highly interesting; novel and biologically convincing locomotion, great abilities and great potential for further evolution.
        
        ═══ OUTPUT FORMAT ═══
        Respond ONLY with valid JSON, no text before or after:

        {output_format}
        """

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

def ask_question_on_video(video_path: str, question: str) -> dict:
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

    try:
        prompt = f"""
                You are looking at video of a 5 second simulation showing a simulated robot composed of a white torso and colored legs. It stands on a green checkered floor, and the background is blue.
                Focus solely on this question and answer it : 
                {question}"""

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

        return text
    finally:
        pass

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

    #question = "Describe the robot morphology and behavior."
    #resp = ask_question_on_video("./video/jumper.mp4", question)
    #print(f"Question : {question} \nAnswer : {resp}")

    resp = score_robot_video("./video/jumper_2.mp4", "jumping as high as possible")
    print("✅ Scores :")
    print(f"  Observation     : {resp.get('observation')}")
    print(f"  Interpretation  : {resp.get('interpretation')}")
    print(f"  Coherence       : {resp.get('coherence')}")
    print(f"  Originality     : {resp.get('originality')}")
    print(f"  Potential        : {resp.get('potential')}")
    print(f"  Overall        : {round((1.0 * float(resp.get('coherence')['score']) + 0.5 * float(resp.get('originality')['score']) + 1.5 * float(resp.get('potential')['score'])) / 3, 0)}")

    # video_path = sys.argv[1] if len(sys.argv) > 1 else "./video/mid.mp4"20
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
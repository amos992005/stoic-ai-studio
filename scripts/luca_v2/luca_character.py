"""
The character, described in exactly one place.

Two rules are encoded here rather than trusted to discipline:

  * No celebrity name ever enters a prompt. Naming a real actor does not just
    invite likeness problems, it *ages the character* — the model reproduces the
    person it has seen in the training data, which is a 50-something. He is
    described by attributes only, and the trigger word `ohwx` carries the
    identity once the LoRA is trained.
  * Hands are never near the face, and framing is explicit. Face/hand contact
    is where diffusion models produce the artefacts that make a carousel look
    synthetic.
"""

from __future__ import annotations

from typing import Dict, List

#: The LoRA trigger word. Rare by construction, so it cannot collide with a real
#: token, and stable across the dataset, training and inference.
TRIGGER = "ohwx"

#: The age the character is, and stays. Everything downstream is anchored to it.
AGE = 27

#: The canonical description. Used verbatim for the master and the dataset, and
#: as the prefix for every carousel image, so identity language never drifts.
SUBJECT = (
    f"{TRIGGER} 27-year-old Italian man, athletic build, sun-kissed dirty blonde "
    "tousled hair, piercing blue-green eyes, sharp jawline, light 3-day stubble"
)

#: Qualities that make an image read as a real photograph rather than a render.
PHOTO_REALISM = (
    "candid 24mm iPhone 16 Pro photo, natural sunlight, skin pores, "
    "unretouched, photorealistic, sharp focus on the eyes"
)

#: Never rendered. Negative prompt only.
NEGATIVE_PROMPT = (
    "cartoon, illustration, 3d render, cgi, painting, anime, "
    "airbrushed skin, plastic skin, waxy skin, doll-like, "
    "deformed hands, extra fingers, fused fingers, missing fingers, "
    "hands near face, hand covering face, face touching, "
    "mutated, distorted, disfigured, bad anatomy, bad proportions, "
    "extra limbs, long neck, asymmetric eyes, crossed eyes, "
    "oversaturated, hdr, oversharpened, watermark, text, logo, signature, "
    "blurry, low resolution, jpeg artifacts, grainy, "
    "old, elderly, aged, wrinkles, grey hair, balding, "
    "car door clipping, cropped head, out of frame"
)

#: A copyright-safe note for the metrics: what we deliberately do not say.
FORBIDDEN_TERMS = ("brad pitt", "brad", "pitt", "bradley")


def check_prompt(text: str) -> str:
    """Raise if a forbidden name leaked into a prompt. Called before generating."""
    lowered = text.lower()
    for term in FORBIDDEN_TERMS:
        if term in lowered:
            raise ValueError(f"forbidden name {term!r} in prompt: {text[:120]!r}")
    return text


# --------------------------------------------------------------------------- #
# Dataset — 15 varied photographs of the same person
# --------------------------------------------------------------------------- #
#: (name, prompt, seed). Variety is the point: if all fifteen are close-ups he
#: learns one crop, and the carousel inherits that framing.
#:
#: `framing` is spelled out per image — "close-up", "medium shot", "full body" —
#: because that is what the LoRA actually learns about composition, and what
#: keeps a subject from being clipped by a car door later.
DATASET: List[Dict[str, object]] = [
    {"name": "train_01_front_closeup",
     "framing": "close-up portrait, facing camera",
     "lighting": "soft window light from the left",
     "expression": "relaxed neutral expression", "seed": 1001},
    {"name": "train_02_three_quarter",
     "framing": "head and shoulders, three-quarter view turned to his right",
     "lighting": "warm late-afternoon sun, slight rim light",
     "expression": "subtle closed-mouth smile", "seed": 1002},
    {"name": "train_03_profile",
     "framing": "head and shoulders, near-profile looking off to the left",
     "lighting": "hard directional sunlight from the right",
     "expression": "serious, jaw set", "seed": 1003},
    {"name": "train_04_medium_smile",
     "framing": "medium shot from the chest up, facing camera",
     "lighting": "bright open shade",
     "expression": "genuine wide smile showing teeth", "seed": 1004},
    {"name": "train_05_medium_street",
     "framing": "medium shot, walking toward camera",
     "lighting": "overcast diffused daylight",
     "expression": "laughing, looking slightly off camera", "seed": 1005},
    {"name": "train_06_full_body",
     "framing": "full body shot, standing, entire figure in frame",
     "lighting": "golden hour backlight",
     "expression": "confident neutral stance", "seed": 1006},
    {"name": "train_07_low_angle",
     "framing": "medium shot from a low angle looking up",
     "lighting": "harsh midday sun with strong shadows",
     "expression": "challenging half-smile", "seed": 1007},
    {"name": "train_08_high_angle",
     "framing": "close-up from slightly above, looking up at the camera",
     "lighting": "soft light from above",
     "expression": "curious, eyebrows slightly raised", "seed": 1008},
    {"name": "train_09_cafe_interior",
     "framing": "medium shot seated at a table",
     "lighting": "warm tungsten interior light, shallow depth of field",
     "expression": "thoughtful, looking away", "seed": 1009},
    {"name": "train_10_beach_daylight",
     "framing": "medium shot on a beach, sea behind",
     "lighting": "bright direct sun, high contrast",
     "expression": "squinting slightly into the sun, easy smile", "seed": 1010},
    {"name": "train_11_evening_portrait",
     "framing": "close-up portrait, head turned fifteen degrees",
     "lighting": "blue hour ambient with warm practical light",
     "expression": "calm, direct gaze", "seed": 1011},
    {"name": "train_12_studio_neutral",
     "framing": "head and shoulders against a plain mid-grey backdrop",
     "lighting": "even studio softbox lighting",
     "expression": "neutral catalog expression", "seed": 1012},
    {"name": "train_13_over_shoulder",
     "framing": "medium shot over one shoulder looking back at the camera",
     "lighting": "late sun raking across the frame",
     "expression": "slight smirk", "seed": 1013},
    {"name": "train_14_hat_optional",
     "framing": "medium shot, hands in pockets, shoulders relaxed",
     "lighting": "diffused city light",
     "expression": "warm half-smile", "seed": 1014},
    {"name": "train_15_tight_face",
     "framing": "very tight close-up on the eyes and jaw",
     "lighting": "dramatic side light, deep shadow on one side",
     "expression": "intense direct gaze", "seed": 1015},
]


def dataset_prompt(entry: Dict[str, object]) -> str:
    """The full prompt for one training photograph."""
    return check_prompt(
        f"{SUBJECT}, {entry['framing']}, {entry['lighting']}, "
        f"{entry['expression']}, no hands near the face, "
        "plain everyday clothing, simple uncluttered background, "
        f"{PHOTO_REALISM}"
    )


# --------------------------------------------------------------------------- #
# Carousel — four finished slides
# --------------------------------------------------------------------------- #
#: The mandatory prompt rules, per the brief:
#:   * `medium full shot` and `full body` framing, so a car or a railing cannot
#:     crop him in half;
#:   * `candid 24mm iPhone 16 Pro photo`, `natural sunlight`, `skin pores` on
#:     every slide;
#:   * no hands near the face.
CAROUSEL: List[Dict[str, object]] = [
    # Slide 01 is the master portrait itself, re-rendered through the LoRA.
    # It keeps this exact filename and framing because it is also the source
    # image SadTalker animates in process 2: a medium full shot on a yacht makes
    # a poor talking head, and a portrait makes the right one. Keeping the two
    # identical is what makes either version a valid avatar source.
    {"file": "01_luca_master.png",
     "seed": 42, "size": (1024, 1024),
     "scene": "close-up portrait facing the camera, plain white linen shirt, "
              "blurred Mediterranean harbour behind him",
     "expression": "relaxed neutral expression, soft window light"},
    {"file": "02_luca_yacht.png",
     "seed": 43, "size": (832, 1024),
     "scene": "full body shot standing barefoot at the bow of the same yacht, "
              "Mediterranean sea and coastline behind, one hand resting on the "
              "railing at chest height and the other holding sunglasses at his "
              "side, both hands clearly visible with five natural fingers",
     "expression": "looking off toward the horizon, relaxed"},
    {"file": "03_luca_porsche.png",
     "seed": 44, "size": (832, 1024),
     "scene": "medium full shot leaning against the front fender of a parked "
              "white Porsche 911 GT3 RS on a cliffside road above the Amalfi "
              "coast, whole car door and both legs in frame, no part of him "
              "clipped by the car, arms folded loosely across his chest with "
              "both hands visible and five natural fingers",
     "expression": "sunglasses pushed up into his hair, easy confidence"},
    {"file": "04_luca_party.png",
     "seed": 45, "size": (832, 1024),
     "scene": "medium full shot at an exclusive Ibiza rooftop party at night, "
              "warm string lights and blurred crowd behind, wearing an open "
              "black silk shirt, holding a whiskey glass at chest height in one "
              "hand, both hands clearly visible with five natural fingers",
     "expression": "relaxed half-smile, direct on-camera flash"},
]


def carousel_prompt(entry: Dict[str, object]) -> str:
    """The full prompt for one carousel image."""
    return check_prompt(
        f"{SUBJECT}, {entry['scene']}, {entry['expression']}, "
        f"no hands near the face, {PHOTO_REALISM}"
    )

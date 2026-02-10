import os
from huggingface_hub import InferenceClient

import os

HF_TOKEN = os.getenv("HF_TOKEN")


client = InferenceClient(
    model="google/flan-t5-small",
    token=HF_TOKEN
)

def generate_description(action, confidence):
    prompt = (
        f"Write one short sentence describing a video where "
        f"the action is {action}. Confidence is {confidence:.2f}."
    )

    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=40
        )
        return response.strip()

    except Exception as e:
        print("⚠️ HuggingFace description failed, using fallback.")
        # Fallback descriptions (SAFE & FAST)
        fallback = {
            "CricketShot": "A person plays a cricket shot by swinging the bat.",
            "Punch": "A person throws a punch.",
            "PlayingCello": "A person is playing a cello.",
            "ShavingBeard": "A person is shaving their beard.",
            "TennisSwing": "A person swings a tennis racket."
        }
        return fallback.get(action, f"A video showing the action {action}.")

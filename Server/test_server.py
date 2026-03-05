# test_service.py
# A simple test script for LayoutDPO Reward Service
# Requires: requests, pillow, numpy

import requests
import random
import string
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Configuration
SERVICE_URL = "http://localhost:37777"
REWARD_ENDPOINT = f"{SERVICE_URL}/reward"
PING_ENDPOINT = f"{SERVICE_URL}/ping"

def random_image_b64(width=256, height=256):
    """
    Generate a random RGB image and return it as a base64-encoded JPEG string.
    """
    array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(array)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return b64

def random_prompt(length=12):
    """
    Generate a random alphanumeric string to serve as a dummy prompt.
    """
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))


def test_ping():
    resp = requests.post(PING_ENDPOINT)
    print(f"PING status: {resp.status_code}, body: {resp.json()}")


def test_reward():
    # Prepare random image and meta
    image_b64 = random_image_b64()
    prompt = "chaos"
    meta = {
        "annotations": [
            {
                "prompt": prompt,
                # dummy bbox and point
                "bbox": [0, 0, 1, 1],
            }
        ]
    }

    payload = {
        "image_b64": image_b64,
        "meta": meta
    }

    resp = requests.post(REWARD_ENDPOINT, json=payload)
    print(f"REWARD status: {resp.status_code}")
    try:
        print(f"REWARD body: {resp.json()}")
    except Exception:
        print(f"Non-JSON response: {resp.text}")


if __name__ == "__main__":
    print("Testing LayoutDPO Reward Service...")
    test_ping()
    test_reward()

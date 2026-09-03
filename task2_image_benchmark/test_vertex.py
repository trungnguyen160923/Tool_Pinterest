import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
USE_ENTERPRISE = os.getenv("GOOGLE_GENAI_USE_ENTERPRISE")

print("Project :", PROJECT_ID)
print("Location:", LOCATION)
print("Enterprise backend:", USE_ENTERPRISE)

if not PROJECT_ID:
    raise RuntimeError("Thiếu GOOGLE_CLOUD_PROJECT trong .env")

if str(USE_ENTERPRISE).lower() != "true":
    raise RuntimeError("GOOGLE_GENAI_USE_ENTERPRISE phải là True")


client = genai.Client()

input_path = BASE_DIR / "dataset" / "img2.jpg"

if not input_path.exists():
    raise FileNotFoundError(
        f"Không tìm thấy ảnh: {input_path}"
    )

input_image = Image.open(input_path).convert("RGB")

prompt = """
Enhance and upscale this image while preserving the original image
as faithfully as possible.

Preserve composition, people, objects, geometry, colors, textures,
lighting, background, depth of field, and all existing visual elements.

Improve sharpness and fine detail without inventing unsupported details.

Do not add, remove, redesign, beautify, move, crop, or replace anything.
""".strip()


print()
print("Calling Nano Banana 2 Lite...")
print("Model: gemini-3.1-flash-lite-image")
print("Target: 1K")
print()


response = client.models.generate_content(
    model="gemini-3.1-flash-lite-image",
    contents=[
        input_image,
        prompt,
    ],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="2:3",
            image_size="1K",
            output_mime_type="image/jpeg",
        ),
    ),
)


output_path = BASE_DIR / "vertex_smoke_test.jpg"

saved = False

for part in response.parts:
    if part.inline_data is not None:
        raw = part.inline_data.data

        output_path.write_bytes(raw)

        saved = True
        break


if not saved:
    raise RuntimeError(
        "Cloud backend không trả về image output."
    )


with Image.open(output_path) as generated:
    print("SUCCESS")
    print("Output:", output_path)
    print(
        "Size:",
        f"{generated.width}x{generated.height}"
    )


usage = response.usage_metadata

if usage is not None:
    print()
    print("Usage metadata:")
    print(
        "Prompt tokens:",
        getattr(usage, "prompt_token_count", None),
    )
    print(
        "Output tokens:",
        getattr(usage, "candidates_token_count", None),
    )
    print(
        "Total tokens:",
        getattr(usage, "total_token_count", None),
    )
    
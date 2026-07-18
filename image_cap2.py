from pathlib import Path
from PIL import Image
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)

# --------------------------------------------------
# Load Image
# --------------------------------------------------

IMAGE_PATH = Path(__file__).parent / "image.jpg"
image = Image.open(IMAGE_PATH).convert("RGB")

# --------------------------------------------------
# Captioning Model
# --------------------------------------------------

caption_processor = AutoProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# --------------------------------------------------
# VQA Model
# --------------------------------------------------

vqa_processor = AutoProcessor.from_pretrained(
    "Salesforce/blip-vqa-base"
)

vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
)


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def image_caption(image):
    """Generate an image caption."""

    inputs = caption_processor(images=image, return_tensors="pt")

    outputs = caption_model.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        early_stopping=True,
    )

    return caption_processor.decode(outputs[0], skip_special_tokens=True)


def conditional_caption(image, prompt):
    """Generate a caption conditioned on a text prompt."""

    inputs = caption_processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    outputs = caption_model.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        early_stopping=True,
    )

    return caption_processor.decode(outputs[0], skip_special_tokens=True)


def answer_question(image, question):
    """Answer a question about an image."""

    inputs = vqa_processor(
        images=image,
        text=question,
        return_tensors="pt",
    )

    outputs = vqa_model.generate(
        **inputs,
        max_length=20,
    )

    return vqa_processor.decode(outputs[0], skip_special_tokens=True)


# --------------------------------------------------
# Demonstration
# --------------------------------------------------

print("=" * 70)
print("1. IMAGE CAPTIONING")
print("=" * 70)

print(image_caption(image))


print("\n" + "=" * 70)
print("2. CONDITIONAL IMAGE CAPTIONING")
print("=" * 70)

prompt = "A photo of"

print("Prompt :", prompt)
print("Output :", conditional_caption(image, prompt))


print("\n" + "=" * 70)
print("3. VISUAL QUESTION ANSWERING")
print("=" * 70)

questions = [
    "What animal is shown?",
    "What is the cat doing?",
    "Where is the cat?",
]

for question in questions:
    print(f"\nQ: {question}")
    print(f"A: {answer_question(image, question)}")
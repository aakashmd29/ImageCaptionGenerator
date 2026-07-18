import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

# --------------------------------------------------
# Load the model once during application startup
# --------------------------------------------------

MODEL_NAME = "Salesforce/blip-image-captioning-base"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)


# --------------------------------------------------
# Caption Generation
# --------------------------------------------------

def generate_caption(image: np.ndarray) -> str:
    """
    Generates a caption for the given image.

    Args:
        image: Input image as a NumPy array (received from Gradio).

    Returns:
        Generated image caption.
    """

    if image is None:
        return "Please upload an image."
    
    # Convert NumPy array -> PIL Image
    pil_image = Image.fromarray(image).convert("RGB")

    return generate_caption_for_pil_image(pil_image)

def generate_caption_for_pil_image(image):
    if image is None:
        return "Please upload an image."

    # Preprocess the image
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Generate caption
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            early_stopping=True
        )

    # Decode prediction
    caption = processor.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return caption

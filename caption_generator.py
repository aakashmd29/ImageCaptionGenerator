import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

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

    # Convert NumPy array -> PIL Image
    pil_image = Image.fromarray(image).convert("RGB")

    # Preprocess the image
    inputs = processor(
        images=pil_image,
        return_tensors="pt"
    )

    # Generate caption
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
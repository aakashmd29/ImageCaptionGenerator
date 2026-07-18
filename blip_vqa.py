import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BlipForQuestionAnswering

VQA_MODEL_NAME = "Salesforce/blip-vqa-base"
MAX_ANSWER_LENGTH = 20

vqa_processor = AutoProcessor.from_pretrained(VQA_MODEL_NAME)
vqa_model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL_NAME)


def answer_question(image: np.ndarray, question: str) -> str:
    """
    Answer a question about an image.

    Args:
        image: Input image as a NumPy array.
        question: Natural language question.

    Returns:
        The generated answer.
    """

    if image is None:
        return "Please upload an image."

    pil_image = Image.fromarray(image).convert("RGB")

    inputs = vqa_processor(
        images=pil_image,
        text=question,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = vqa_model.generate(
            **inputs,
            max_length=MAX_ANSWER_LENGTH,
        )

    return vqa_processor.decode(
        outputs[0],
        skip_special_tokens=True,
    )

import gradio as gr
import numpy as np
from caption_generator import generate_caption

gradio_interface = gr.Interface(
    fn=generate_caption, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Caption Generator",
    description="This is a simple web app for generating captions for images using a trained model."
)

gradio_interface.launch()


import os
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from hair_swap import HairFast, get_parser

def download_models():
    """Download required model checkpoints if they don't exist"""
    os.makedirs("pretrained_models", exist_ok=True)
    
    # Create necessary directories
    for dir_name in ["StyleGAN", "Rotate", "Blending", "PostProcess"]:
        os.makedirs(f"pretrained_models/{dir_name}", exist_ok=True)
    
    # Download checkpoints if they don't exist
    if not os.path.exists("pretrained_models/StyleGAN/ffhq.pt"):
        os.system("gdown https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT -O pretrained_models/StyleGAN/ffhq.pt")
    if not os.path.exists("pretrained_models/Rotate/rotate_best.pth"):
        os.system("gdown https://drive.google.com/uc?id=1-0QHvYT3y3Ph6Z4ZlFt9H1ZpQZHYUJgp -O pretrained_models/Rotate/rotate_best.pth")
    if not os.path.exists("pretrained_models/Blending/checkpoint.pth"):
        os.system("gdown https://drive.google.com/uc?id=1-4sFJHG9U0GpnY1F1Ys9Zc2X9P2ZQ9Qp -O pretrained_models/Blending/checkpoint.pth")
    if not os.path.exists("pretrained_models/PostProcess/pp_model.pth"):
        os.system("gdown https://drive.google.com/uc?id=1-7JfG7X6Y7X6Y7X6Y7X6Y7X6Y7X6Y7X6 -O pretrained_models/PostProcess/pp_model.pth")

def predict(face_image, shape_image, color_image, mixing=0.95, smooth=5):
    """
    Run HairFastGAN prediction
    
    Parameters:
    - face_image: PIL Image of the face
    - shape_image: PIL Image of the desired hair shape
    - color_image: PIL Image of the desired hair color
    - mixing: float, hair blending parameter (default: 0.95)
    - smooth: int, dilation and erosion parameter (default: 5)
    
    Returns:
    - PIL Image of the result
    """
    # Download models if needed
    download_models()
    
    # Set up model arguments
    model_args = get_parser().parse_args([])
    model_args.mixing = mixing
    model_args.smooth = smooth
    model_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model
    hair_fast = HairFast(model_args)
    
    # Process images
    result = hair_fast.swap(
        face_image,
        shape_image,
        color_image,
        align=True  # Enable face alignment for better results
    )
    
    # Convert result to PIL Image
    if isinstance(result, tuple):
        result = result[0]  # Get the first tensor if multiple are returned
    
    result = result.cpu().clamp(0, 1)
    result = (result * 255).to(torch.uint8)
    result = result.permute(1, 2, 0).numpy()
    result = Image.fromarray(result)
    
    return result 
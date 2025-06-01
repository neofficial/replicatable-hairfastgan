from cog import BasePredictor, Input, Path
from PIL import Image
import torch
from hair_swap import HairFast, get_parser

def download_models():
    import os
    os.makedirs("pretrained_models", exist_ok=True)
    for dir_name in ["StyleGAN", "Rotate", "Blending", "PostProcess"]:
        os.makedirs(f"pretrained_models/{dir_name}", exist_ok=True)
    if not os.path.exists("pretrained_models/StyleGAN/ffhq.pt"):
        os.system("gdown https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT -O pretrained_models/StyleGAN/ffhq.pt")
    if not os.path.exists("pretrained_models/Rotate/rotate_best.pth"):
        os.system("gdown https://drive.google.com/uc?id=1-0QHvYT3y3Ph6Z4ZlFt9H1ZpQZHYUJgp -O pretrained_models/Rotate/rotate_best.pth")
    if not os.path.exists("pretrained_models/Blending/checkpoint.pth"):
        os.system("gdown https://drive.google.com/uc?id=1-4sFJHG9U0GpnY1F1Ys9Zc2X9P2ZQ9Qp -O pretrained_models/Blending/checkpoint.pth")
    if not os.path.exists("pretrained_models/PostProcess/pp_model.pth"):
        os.system("gdown https://drive.google.com/uc?id=1-7JfG7X6Y7X6Y7X6Y7X6Y7X6Y7X6Y7X6 -O pretrained_models/PostProcess/pp_model.pth")

class Predictor(BasePredictor):
    def setup(self):
        download_models()
        model_args = get_parser().parse_args([])
        model_args.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hair_fast = HairFast(model_args)

    def predict(
        self,
        face_image: Path = Input(description="Face image"),
        shape_image: Path = Input(description="Shape reference image"),
        color_image: Path = Input(description="Color reference image"),
        mixing: float = Input(default=0.95, description="Hair blending parameter"),
        smooth: int = Input(default=5, description="Dilation and erosion parameter"),
    ) -> Path:
        # Load images
        face = Image.open(str(face_image)).convert("RGB")
        shape = Image.open(str(shape_image)).convert("RGB")
        color = Image.open(str(color_image)).convert("RGB")

        # Set parameters
        self.hair_fast.args.mixing = mixing
        self.hair_fast.args.smooth = smooth

        # Run model
        result = self.hair_fast.swap(face, shape, color, align=True)
        if isinstance(result, tuple):
            result = result[0]
        result = result.cpu().clamp(0, 1)
        result = (result * 255).to(torch.uint8)
        result = result.permute(1, 2, 0).numpy()
        out_img = Image.fromarray(result)
        out_path = "/tmp/output.png"
        out_img.save(out_path)
        return Path(out_path)
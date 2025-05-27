# HairFastGAN on Replicate

This repository contains the necessary files to run HairFastGAN on Replicate.

## Model Description

HairFastGAN is a deep learning model for hairstyle transfer. It can transfer both the shape and color of hair from reference images to a target face image.

## Usage on Replicate

To use this model on Replicate, you'll need to:

1. Create a new model on Replicate
2. Push this repository to GitHub
3. Connect your GitHub repository to Replicate
4. Deploy the model

### Input Parameters

The model accepts the following inputs:

- `face_image`: The target face image to apply the hairstyle to
- `shape_image`: The reference image for the desired hair shape
- `color_image`: The reference image for the desired hair color
- `mixing` (optional): Hair blending parameter (default: 0.95)
- `smooth` (optional): Dilation and erosion parameter (default: 5)

### Output

The model returns a single image with the transferred hairstyle applied to the target face.

## Local Development

To test the model locally before deploying to Replicate:

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the model:
```python
from predict import predict
from PIL import Image

# Load your images
face = Image.open("path/to/face.jpg")
shape = Image.open("path/to/shape.jpg")
color = Image.open("path/to/color.jpg")

# Run prediction
result = predict(face, shape, color)
result.save("output.jpg")
```

## Model Checkpoints

The model requires several pretrained checkpoints that will be automatically downloaded when running the model:

- StyleGAN FFHQ model
- Rotate model
- Blending model
- Post-processing model

These will be downloaded automatically when the model is first run.

## License

This model is released under the same license as the original HairFastGAN repository. 
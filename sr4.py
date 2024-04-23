import time
import torch
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
from PIL import Image
import requests
import numpy as np

# Load the image processor and model from pretrained weights
processor = AutoImageProcessor.from_pretrained(
    "caidas/swin2SR-classical-sr-x4-64")
model = Swin2SRForImageSuperResolution.from_pretrained(
    "caidas/swin2SR-classical-sr-x4-64")

# Load image from a URL
# image = Image.open(requests.get("https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/butterfly.jpg", stream=True).raw)
file = 'input6.png'
image = Image.open(file)

# Prepare image for the model
start = time.time()
inputs = processor(image, return_tensors="pt")
# Forward pass without updating gradients
with torch.no_grad():
    outputs = model(**inputs)
# Process output
output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
output = np.moveaxis(output, source=0, destination=-1)
# Convert from float32 to uint8
output_image = (output * 255.0).round().astype(np.uint8)
print("Time Taken: %f" % (time.time() - start))

# # Save the enhanced image

# Record the original dimensions
original_width, original_height = image.size

# Calculate target dimensions (original dimensions * 4)
target_width = original_width * 4
target_height = original_height * 4

# Crop the super-resolution image to the target size
# Ensuring the dimensions do not exceed the current image size
crop_width = min(target_width, output_image.shape[1])
crop_height = min(target_height, output_image.shape[0])
cropped_image = output_image[:crop_height, :crop_width]

# Save the enhanced and cropped image
enhanced_image = Image.fromarray(cropped_image)
enhanced_image.save(file + '-sr4.png')

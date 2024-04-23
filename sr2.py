import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
# import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def preprocess_image(image_path):
    """Loads image from path and preprocesses to make it model ready with padding."""
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    if hr_image.shape[-1] == 4:  # Remove alpha channel if present
        hr_image = hr_image[..., :-1]

    original_size = hr_image.shape[:2]  # Save original dimensions

    # Calculate the padding sizes needed to make height and width multiple of 4
    height_pad = 4 - hr_image.shape[0] % 4 if hr_image.shape[0] % 4 != 0 else 0
    width_pad = 4 - hr_image.shape[1] % 4 if hr_image.shape[1] % 4 != 0 else 0

    # Pad the image using the 'REFLECT' mode which duplicates the edge pixels
    # 'CONSTANT' mode can be used if zero padding is desired
    # 'SYMMETRIC' mode reflects the image around the edge
    hr_image = tf.pad(hr_image, [[0, height_pad], [
                      0, width_pad], [0, 0]], mode='REFLECT')

    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0), original_size


def save_image(image, filename, original_size):
    """
      Saves unscaled Tensor Images.
      Args:
        image: 3D image tensor. [height, width, channels]
        filename: Name of the file to save.
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        # Calculate the target size as original_size * scale_factor
        # Assuming scale_factor is 4
        scale_factor = 4
        target_height = original_size[0] * scale_factor
        target_width = original_size[1] * scale_factor

        # Crop the image to the target size
        # Calculate the starting points for cropping (center crop)
        start_y = (image.shape[0] - target_height) // 2
        start_x = (image.shape[1] - target_width) // 2
        image = tf.image.crop_to_bounding_box(
            image, start_y, start_x, target_height, target_width)

        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(filename)
    print("Saved as %s" % filename)


model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

file = 'input6.png'
hr_image, hr_image_size = preprocess_image(file)
start = time.time()
fake_image = model(hr_image)
fake_image = tf.squeeze(fake_image)
print("Time Taken: %f" % (time.time() - start))

save_image(tf.squeeze(fake_image), file+'-sr2.png', hr_image_size)

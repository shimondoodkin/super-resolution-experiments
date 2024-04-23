import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

# List TensorFlow GPU devices and set memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Set memory growth for the first GPU device
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Memory growth set on GPU 0")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def load_and_pad_image(image_path, patch_size=4):
    """Load image and pad only on the right and bottom to make dimensions fit the patch size."""
    hr_image = tf.image.decode_image(tf.io.read_file(image_path), channels=3)
    # Save the original size for later cropping
    original_size = hr_image.shape[:2]

    # Calculate padding sizes to make height and width multiples of patch_size, but only pad right and bottom
    height_pad = (patch_size - hr_image.shape[0] % patch_size) % patch_size
    width_pad = (patch_size - hr_image.shape[1] % patch_size) % patch_size

    # Apply padding to the bottom and right sides of the image
    hr_image = tf.pad(hr_image, [[0, height_pad], [
                      0, width_pad], [0, 0]], mode='REFLECT')
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0), original_size


def save_image(image, filename, original_size):
    """Save the processed image, cropping to the original dimensions scaled by 4."""
    # Crop the image to the original dimensions multiplied by 4
    target_height = original_size[0] * 4
    target_width = original_size[1] * 4
    image = tf.image.crop_to_bounding_box(
        image, 0, 0, target_height, target_width)

    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(filename)
    print("Saved as %s" % filename)


# Load the ESRGAN model
model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
file = 'input-b.jpg'

# Preprocess the image and process it


def process_image(model, image):
    """Process the full image using the model without splitting into patches, assuming the model can handle it."""
    return model(image)


def process_image_in_batches(model, image, batch_size=50):
    """Process the image in manageable batches."""
    # Assume image is already expanded in the first dimension
    num_batches = image.shape[1] // batch_size
    results = []
    print('start batching total '+str(num_batches))
    for i in range(num_batches):
        print('running batch '+str(i)+'  done ' + str(i*100.0//num_batches))

        batch = image[:, i*batch_size:(i+1)*batch_size, :, :]
        processed_batch = model(batch)
        results.append(processed_batch)

    # Concatenate all processed batches
    full_image = tf.concat(results, axis=1)
    return full_image


# Preprocess the image and process it in batches
hr_image, original_size = load_and_pad_image(file)
start = time.time()
upscaled_image = process_image_in_batches(model, hr_image)
upscaled_image = tf.squeeze(upscaled_image)
print("Time Taken: %f" % (time.time() - start))

# Save the upscaled and cropped image
save_image(upscaled_image, file + '-sr5.png', original_size)

#
# this file is https://www.beam.cloud/ deployment of https://tfhub.dev/captain-pool/esrgan-tf2/1 
# when you wil sign up you will see quickstart, that will help you setup venv and token. put this file in the quickstart folder and deploy
# 
# deploy command:
#    beam deploy app.py:upscale
#
# see request.py to run this like
#    python.exe request.py
#

from beam import Image, Volume, endpoint, Output, env

# Since these packages are only installed remotely on Beam, this block ensures the interpreter doesn't try to import them locally
if env.is_remote():
    import tensorflow_hub as hub
    import tensorflow as tf
    import numpy as np
    import base64
    from PIL import Image as PILImage
    import io
    import os

# The container image for the remote runtime
image = (
    Image(python_version="python3.9")
    .add_python_packages([
        "tensorflow[and-cuda]",
        "tensorflow-hub",
        "numpy",
        "pillow"
    ])
)

# Load ESRGAN model
MODEL_URL = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def load_model():
    os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as e:
            print(e)
    return hub.load(MODEL_URL)

def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return PILImage.open(io.BytesIO(image_data)).convert("RGB") # Ensure RGB

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def preprocess_image(image, patch_size=4):
    hr_image = tf.convert_to_tensor(np.array(image))
    original_size = hr_image.shape[:2]
    height_pad = (patch_size - hr_image.shape[0] % patch_size) % patch_size
    width_pad = (patch_size - hr_image.shape[1] % patch_size) % patch_size
    hr_image = tf.pad(hr_image, [[0, height_pad], [0, width_pad], [0, 0]], mode='REFLECT')
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0), original_size


def process_image_in_batches(model, image, batch_size=128): # Adjusted batch size
    """Process the image in manageable batches."""
    # Assume image is already expanded in the first dimension
    num_batches = image.shape[1] // batch_size
    results = []

    for i in range(num_batches):
        batch = image[:, i*batch_size:(i+1)*batch_size, :, :]
        processed_batch = model(batch)
        results.append(processed_batch)

    # Handle the remaining part of the image if the size is not divisible by the batch size
    if image.shape[1] % batch_size != 0:
        remaining_batch = image[:, num_batches*batch_size:, :, :]
        processed_remaining_batch = model(remaining_batch)
        results.append(processed_remaining_batch)

    # Concatenate all processed batches
    full_image = tf.concat(results, axis=1)
    return full_image

def upscale_image(model, image):
    #return model(image) # removed this line
    return process_image_in_batches(model, image) # added batch processing

def crop_image(image, original_size):
    target_height = original_size[0] * 4
    target_width = original_size[1] * 4
    return tf.image.crop_to_bounding_box(image, 0, 0, target_height, target_width)

def upscale_base64_image(model, base64_str):
    image = base64_to_image(base64_str)
    hr_image, original_size = preprocess_image(image)
    upscaled_image = upscale_image(model, hr_image)
    upscaled_image = tf.squeeze(upscaled_image)
    upscaled_image = crop_image(upscaled_image, original_size)
    upscaled_image = tf.clip_by_value(upscaled_image, 0, 255)
    upscaled_image = PILImage.fromarray(tf.cast(upscaled_image, tf.uint8).numpy())
    return image_to_base64(upscaled_image)

@endpoint(	
    name="esrgan-upscale",
    image=image,
    on_start=load_model,
    keep_warm_seconds=5,
    cpu=1,
    memory="16Gi",
    gpu="A10G",
#,
#    cpu=2,
#    memory="16Gi",
#    gpu="T4",

# T4 (16Gi)
# A10G (24Gi)
# A100-40 (40Gi)
# RTX4090 (24Gi)

)
def upscale(context, image_base64: str):

    # Show available GPUs
    gpus = tf.config.list_physical_devices("GPU")

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

    print("🚧 Is built with CUDA:", tf.test.is_built_with_cuda())
    print("🚧 Is GPU available:", tf.test.is_gpu_available())
    print("🚧 GPUs available:", tf.config.list_physical_devices("GPU"))


    model = context.on_start_value
    upscaled_base64 = upscale_base64_image(model, image_base64)
    return {"image_base64": upscaled_base64}

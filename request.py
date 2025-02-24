# see app.py for details
# this file consumes api created with app.py

import requests
import base64

# Beam API details -- make sure to replace with your own credentials
url = 'https://your-url-here.app.beam.cloud'
TOKEN='your token here'

headers = {
    'Connection': 'keep-alive',
    'Content-Type': 'application/json',
    'Authorization': 'Bearer '+TOKEN
}

# Load image and encode it to base64
def load_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

# Save base64 image to file
def save_image_from_base64(image_base64, output_path):
    image_data = base64.b64decode(image_base64)
    with open(output_path, "wb") as image_file:
        image_file.write(image_data)

# Send a POST request to the Beam endpoint
def call_beam_api(image_base64):
    data = {
        "image_base64": image_base64
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

if __name__ == "__main__":
    image_path = "example.png"
    output_path = "output.png"
    image_base64 = load_image_as_base64(image_path)
    result = call_beam_api(image_base64)
    if "image_base64" in result:
        save_image_from_base64(result["image_base64"], output_path)
    else:
        print("no image_base64 in response json")

    # print(result)

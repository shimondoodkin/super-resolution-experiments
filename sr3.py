import time
from super_image import DrlnModel, ImageLoader
from PIL import Image
import requests

# image = Image.open(requests.get('https://paperswithcode.com/media/datasets/Set5-0000002728-07a9793f_zA3bDjj.jpg', stream=True).raw)
file = 'input6.png'
image = Image.open(file)

# scale 2, 3 and 4 models available
model = DrlnModel.from_pretrained('eugenesiow/drln', scale=4)
inputs = ImageLoader.load_image(image)
start = time.time()
preds = model(inputs)
print("Time Taken: %f" % (time.time() - start))

# save the output 2x scaled image to `./scaled_2x.png`
ImageLoader.save_image(preds, file+'-sr3.png')
# save an output comparing the super-image with a bicubic scaling
# ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')

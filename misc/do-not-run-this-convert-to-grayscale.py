import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from PIL import Image

def read_image(img_name):
    img = plt.imread(img_name).astype(float)
    return img

img = read_image('input.png')
img = img[ :, :, 0]

print(img.shape)

imsave('input-grayscale.png', img, cmap='gray')

# im = Image.fromarray(img)
# im.save("input-grayscale.jpeg")
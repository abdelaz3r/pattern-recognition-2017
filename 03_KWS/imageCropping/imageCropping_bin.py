from svgpathtools import svg2paths
from PIL import Image, ImageDraw, ImageOps
import re
import numpy as np
import sys
import os
from skimage import filters

"""
Created on Sunday 14.05.2017

@author: Jerome Treboux
"""

def main():
    # argv[0] : Image and SVG ID to process
    print('Process started....')

    os.makedirs("croppedImages/%s" % sys.argv[1], exist_ok=True)

    im = image_bin()

    # convert to numpy (for convenience)
    im_array = np.asarray(im)

    shape_list, attributes = shape_list_svg()

    print('Image cropping process started...')
    image_cropping(shape_list, im, im_array, attributes)
    print('Process Finished.')


# Image Binarization process
def image_bin():
    im = Image.open('images/%s.jpg' % sys.argv[1])

    im.convert('L')

    bw = np.asarray(im).copy()

    threshold = filters.threshold_otsu(bw)
    bw[bw < threshold] = 0
    bw[bw >= threshold] = 255
    

    # Pixel range is 0...255, 256/2 = 128
    #bw[bw < 128] = 0  # Black
    #bw[bw >= 128] = 255  # White

    # Now we put it back in Pillow/PIL land
    return Image.fromarray(bw)


# Create shape list based on svg paths
def shape_list_svg():
    paths, attributes = svg2paths('ground-truth/locations/%s.svg' % sys.argv[1])

    path_list = []

    for p in attributes:
        path = (re.sub("[^0-9.]", " ", p['d'])).split(" ")
        path = list(filter(None, path))
        path_list.append(list(map(float, path)))

    shape_list = []

    for p in path_list:
        shape = []
        for i in zip(*[iter(p)] * 2):
            shape.append(i)
        shape_list.append(shape)

    return shape_list, attributes


# Image croppring process based on svg paths
def image_cropping(shape_list, im, im_array, attributes):
    count = 0
    for s in shape_list:
        #ImageDraw.Draw(im).polygon(s, outline=1, fill=None)

        # Create mask bin with size of the pic and black fill
        mask_im = Image.new('L', (im_array.shape[1], im_array.shape[0]), 0)
        # Draw the current polygon
        ImageDraw.Draw(mask_im).polygon(s, outline=1, fill=255)
        mask = np.array(mask_im)
        # assemble new image (uint8: 0-255)
        new_im_array = np.empty(im_array.shape, dtype='uint8')

        # colors
        new_im_array[:, :] = im_array[:, :]

        # Between polygon and border, set color to white
        new_im_array[mask == 0] = 255

        new_im = Image.fromarray(new_im_array)
        new_im = new_im.crop(mask_im.getbbox())

        new_im.save("croppedImages/%s/%s.png" % (sys.argv[1], attributes[count]['id']))
        count += 1


if __name__ == "__main__":
    main()

from svgpathtools import svg2paths
from PIL import Image, ImageDraw, ImageOps
import re
import numpy
import sys
import os

# argv[0] : Image and SVG ID to process
print('Process started....')
print('Folder creation....')
os.makedirs(sys.argv[1], exist_ok=True)

im = Image.open('images/%s.jpg' % sys.argv[1]).convert("RGBA")

# convert to numpy (for convenience)
imArray = numpy.asarray(im)

paths, attributes = svg2paths('ground-truth/locations/%s.svg' % sys.argv[1])

path_list = []

for p in attributes:
    path = (re.sub("[^0-9.]", " ",p['d'])).split(" ")
    path = list(filter(None, path))
    path_list.append(list(map(float, path)))

shape_list = []

for p in path_list:
    shape = []
    for i in zip(*[iter(p)]*2):
        shape.append(i)
    shape_list.append(shape)

print('Image cropping process started...')
count = 0
for s in shape_list:
    ImageDraw.Draw(im).polygon(s, outline=1, fill=None)

    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(s, outline=1, fill=1)
    mask = numpy.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = numpy.empty(imArray.shape,dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]

    # transparency (4th column)
    newImArray[:,:,3] = mask*255

    # back to Image from numpy
    newIm = Image.fromarray(newImArray, "RGBA")
    newIm = newIm.crop(maskIm.getbbox())
    #newIm = newIm.convert('1')
    newIm.save("%s/%s.png" % (sys.argv[1], attributes[count]['id']))
    count = count + 1

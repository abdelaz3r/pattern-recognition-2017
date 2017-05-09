# -*- coding: utf-8 -*-
"""
Created on Tue May  9 09:59:41 2017

@author: alvin
"""
from svgpathtools import svg2paths
from PIL import Image
import os, os.path
import glob



#read document images and binarize
image_list = []
svg_list=[]
for filename in glob.glob('path_to-file/*.jpg'): #assuming jpg
    im=Image.open(filename)
    im.convert('1')
    image_list.append(im)





svgs = []
path = "/home/tony/pictures"
valid_images = [".svg"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    svgs.append(open(os.path.join(path,f)))


# svg path extraction
for num in range(len(svgs)):
    paths, attributes = svg2paths(svgs[num])
    # apply paths to images in [num] 

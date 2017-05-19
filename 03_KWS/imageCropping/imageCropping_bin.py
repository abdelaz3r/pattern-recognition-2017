from svgpathtools import svg2paths
from PIL import Image, ImageDraw, ImageOps
import re
import numpy as np
from skimage import filters



def main():
  for i in ['trains','valids']:
    with open("/task/"+i+".txt", "r") as myfile:
        lines = myfile.readlines()
        for line in lines:
            imageid = line.replace("\n","")
            im=Image.open("/images/"+imageid+".jpg")
            
            #each image in training
            #binarize image
            #im = image_bin(im)	
            # convert to numpy (for creating mask)
            im_array = np.asarray(im)		
		
            #get shape and attributes of svg
            shape_list, attributes = shape_list_svg("/ground-truth/locations/"+imageid+".svg")
            print('Processing' +imageid+'...')
                #crop images using shape and attributes
            image_cropping(shape_list, im, im_array, attributes,i)
            print('Process Finished.')
                




# Image Binarization process
def image_bin(im):
    
    im = im.convert('L')

    bw = np.asarray(im).copy()
    threshold = filters.threshold_isodata(bw)
    bw[bw < threshold] = 0
    bw[bw >= threshold] = 255
    
    # Now we put it back in Pillow/PIL land
    return Image.fromarray(bw)


# Create shape list based on svg paths
def shape_list_svg(svg):
    paths, attributes = svg2paths(svg)

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
def image_cropping(shape_list, im, im_array, attributes, des):
    count = 0
    for s in shape_list:
        ImageDraw.Draw(im).polygon(s, outline=1, fill=None)

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
        new_im = image_bin(new_im)
        new_im.save("/"+des+"/%s.png" % (attributes[count]['id']))
        count += 1
#read training document images 
#images_train = {}


if __name__ == "__main__":
    main()

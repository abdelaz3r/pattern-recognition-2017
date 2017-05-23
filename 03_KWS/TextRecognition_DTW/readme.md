# Image cropping based on svg path#

----------
dtw library
https://github.com/honeyext/cdtw

## Requirements ##

- SVG File containing id and paths (in the folder ./ground-truth/locations)
- Image with the same size of the svg (in folder ./images)

## How To ##

Run the script with one parameter "**imageId**". This script will create a new folder for each image to crop. The folder will contain each images, named based on the id of the ground truth.

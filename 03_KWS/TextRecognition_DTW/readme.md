# Image cropping based on svg path#

----------
required dtw library
https://github.com/honeyext/cdtw

## How To ##

- Download and unzip the database and valid files along with the ground truth folder, task, valid
- Place the sripts in the same root folder and call similarclusters.py from the console
- labels are placed for all images in the valid folder and grouped according to their labels, their 
  dismillarity score to the chossen cluster is extarcted and saved in results.txt according to the format specified
## The functions ##
- the imagecropping_bin.py script taes the train and valid document images and extract the various words from
  them placing in train or valid folder as indicated by the task.txt.
  and the ids are placed as the image name as indicated by transcription in the groundtruth
  
- the getdata script contains all the function required to extract the features of images

- similarcluster script , takes the labels of the images from the trascription and calling getdata to extract the 
  features , sotres the features of the image as a pkl file in the folder database, with a loadfeaturestofile function.
  NB#: With this databse folder, there is no need to run get features for the images again.
  the function, initialize_image_clusters will load the features of the images from the pkl file and with the labels from the 
  transcription.txt file , will place similiar images in a cluster with their features, and ids.
- with the image_clusters , the task becomes, to which cluster shd we label a test image, by calling dtw of the test features 
  and the set of images in each cluster, keeping the shortest distance to various clusters and returning the cluster 
  with the minimum distance as the label for the test images, using the evaluation script during which time the cluster is initialized

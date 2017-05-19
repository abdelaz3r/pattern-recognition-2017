# -*- coding: utf-8 -*-
"""
Created on Wed May 17 00:01:25 2017

@author: alvin

"""
import pickle
import glob
from getdata import getfeatures
import os
from cdtw import pydtw

#Load the feature vector from file, if exist if not generate and do dtw
#For the training part
def loadFeatureVector_tofile():
    try:      
        with open("ground-truth/transcription.txt", "r") as myfile:
            lines = myfile.readlines()
            for line in lines:
                #get data in line 
                id_label = line.replace("\n","").split()
                #slipt into id and labels
                im_id = id_label[0]
                label = id_label[1]
                #load feature of said image and store with the label
                feature = getfeatures("train/"+im_id+".png")
                #sava to file data + id
                file = open('database/' + im_id + '.pkl','wb')
                pickle.dump((feature,im_id,label), file)
    except FileNotFoundError:
        print('Training images features saved to DB')
    return 0


# load the features and the label and make a cluster dict based on label
def initialize_image_clusters():
    im_clusters = {}
    #open the images from database
    for filename in glob.glob('database/*'):
        file = open(filename,'rb')
        label_data = pickle.load(file)#tuple with the feature data and label
        #make a dictionary of dictionary(store feature data, and id under label from transcript)
        #but save also the feature data, and the id in a dictionary structure.
        #for every label, dict[label], call the features and do dtw, when least cluster distance is found
        #call the ids of this label cluster
        #####################################TO DO##################
        # Each label contains id and features
        content = [label_data[0],label_data[1]]
        if label_data[2] not in im_clusters:
            im_clusters[label_data[2]] = content
        else:
            im_clusters[label_data[2]].append(content)

    return im_clusters

# load the dtw to classify the test set
def load_test():
    #have to load the features and the labels of the valid too, we use only few
    #by using the appraoch we find the min cluster and all ground truth images will be called. 
    #the task becomes retriving the right cluster for this test image
    with open("ground-truth/transcription.txt", "r") as myfile:
        for filename in glob.glob('valid/*'):
            # get data in line
            file = open(filename, 'rb')
            # slipt into id and labels
            im_id, extension = os.path.splitext(os.path.basename(filename))

            # Get the ground truth
            lines = myfile.readlines()
            for line in lines:
                # get data in line
                id_label = line.replace("\n", "").split()
                # slipt into id and labels
                if id_label[0] == im_id:
                    label = id_label[1]

            # load feature of said image and store with the label
            feature = getfeatures("valid/" + im_id + ".png")

            # start the DTW classification
            for reference in initialize_image_clusters():
                dtw = pydtw.dtw(feature, reference, pydtw.Settings(step='p0sym', window='palival',
                                                                   param=1.0, norm=False, compute_path=True))
                print(dtw.get_dist())


if __name__ == "__main__":
    load_test()
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 00:01:25 2017

@author: alvin

"""
import pickle
from getdata import getfeatures
#Load the feature vector from file, if exist if not generate and do dtw
def loadFeatureVector_tofile():
    try:      
        with open("C:/Users/alvin/Downloads/s2017/PR 2017/ex/PatRec17_KWS_Data-master/ground-truth/transcription.txt", "r") as myfile:
            lines = myfile.readlines()
            for line in lines:
                #get data in line 
                id_label = line.replace("\n","").split()
                #slipt into id and labels
                im_id = id_label[0]
                label = id_label[1]
                #load feature of said image and store with the label
                feature = getfeatures("C:/Users/alvin/Downloads/s2017/PR 2017/ex/PatRec17_KWS_Data-master/train/"+im_id+".png")
                #sava to file data + id
                file = open('C:/Users/alvin/Downloads/s2017/PR 2017/ex/PatRec17_KWS_Data-master/database/' + im_id + '.pkl','wb')
                pickle.dump((feature,im_id,label), file)
    except FileNotFoundError:
        print('Training images features saved to DB')
    return 0


def initialize_image_clusters():#load the features and the label and make a cluster dict based on label
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
        im_clusters[label_data[2]].append(label_data[0],label_data[1])
    
    return dictofdict
    
def load_test(): 
    #have to load the features and the labels of the valid too, we use only few
    #by using the appraoch we find the min cluster and all ground truth images will be called. 
    #the task becomes retriving the right cluster for this test image
    
    

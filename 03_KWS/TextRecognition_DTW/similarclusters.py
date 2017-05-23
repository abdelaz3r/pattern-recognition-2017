# -*- coding: utf-8 -*-
"""
Created on Wed May 17 00:01:25 2017

@author: alvin

"""
import pickle
import glob
from cdtw import pydtw
from getdata import getfeatures
import os
import numpy as np
from sklearn import preprocessing
from operator import itemgetter
#Load the feature vector from file, if exist if not generate and do dtw
def loadFeatureVector_tofile():
    min_max_scaler = preprocessing.MinMaxScaler() 
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
                feature= min_max_scaler.fit_transform(feature)
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
        
        if label_data[2] not in im_clusters:
            im_clusters[label_data[2]] = {}
            im_clusters[label_data[2]]['id']=[]
            im_clusters[label_data[2]]['features']= []
            
            im_clusters[label_data[2]]['id'].append(label_data[1])            
            im_clusters[label_data[2]]['features'].append(label_data[0])
        else:
            im_clusters[label_data[2]]['id'].append(label_data[1])
            im_clusters[label_data[2]]['features'].append(label_data[0])

    return im_clusters

    
# load the dtw to classify the test set
def test_match():
    min_max_scaler = preprocessing.MinMaxScaler() 
    #should be moved to improve performance
    clusters = initialize_image_clusters()
    matches = []
    with open("/ground-truth/transcription.txt", "r") as myfile:
        lines = myfile.readlines()
        for filename in glob.glob('/valid/*'):
    
            # get data in line
            #file = open(filename, 'rb')
            # slipt into id and labels(get id of image in valid)
            im_id, extension = os.path.splitext(os.path.basename(filename))
    
            # Get the ground truth
            for line in lines:
    
                # get data in line
                id_label = line.replace("\n", "").split()
                # slipt into id and labels(use id to get the label)
                if id_label[0] == im_id:
                    label = id_label[1]#for evaluation
                    print ('test',label)
            # load feature matrix of said image 
            print('test',im_id)
            feature = getfeatures("/valid/" + im_id + ".png")
            feature= min_max_scaler.fit_transform(feature)
            # start the DTW classification, put cost to cluster in a cost table
            cost_table = {}
            cost_list = []
            for label_clust in clusters:
                for feature_clust in clusters[label_clust]['features']:
                    dtw = pydtw.dtw(feature.flatten(), feature_clust.flatten(), pydtw.Settings(dist = 'euclid',
                            step='p0sym', 
                            window='palival',
                            param=0.1, 
                            norm=True, 
                            compute_path=False))
                    cost_list.append(dtw.get_dist())#cost of all features in a cluster
                m = np.min(cost_list);
                cost_table[label_clust]= (m)#seems to work sometimes,len([num for num in cost_list if num  < m+s]) )
            
            print('match',min(cost_table, key = cost_table.get))
            print('top 4 ', sorted(cost_table.items(),key=itemgetter(1),reverse=False)[0:4])
            try:
                print('true dist',cost_table[label])
            except KeyError:
                print('non existing template for test')
                
            matches.append(min(cost_table, key = cost_table.get))
        return matches
   

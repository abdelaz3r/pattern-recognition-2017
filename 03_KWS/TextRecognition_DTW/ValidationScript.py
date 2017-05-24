import similarclusters as sc
import getdata
import glob
from sklearn import preprocessing
from cdtw import pydtw
import numpy as np

def main():
     print("Validation process starting...")
    sc.loadFeatureVector_tofile()
    print("All features loaded...")

    clusters = sc.initialize_image_clusters()
    results_cluster = {}
    min_max_scaler = preprocessing.MinMaxScaler()
    
    # For each images to classify and have the dissimilarity
    for filename in glob.glob('/valid/*'):
        file_id = filename.split(".png")
        #print('testing ', file_id)
        
    
        # Get the feature of the image in the 305-309 folder
        feature = getdata.getfeatures(filename)
    
        feature = min_max_scaler.fit_transform(feature)
        # start the DTW classification, put cost to cluster in a cost tabla
        cost_table = {}
        cost_list = []
        for label_clust in clusters:
            for feature_clust in clusters[label_clust]['features']:
                dtw = pydtw.dtw(feature.flatten(), feature_clust.flatten(), pydtw.Settings(dist='euclid',
                                                                                           step='p0sym',
                                                                                           window='palival',
                                                                                           param=0.1,
                                                                                           norm=True,
                                                                                           compute_path=False))
                cost_list.append(dtw.get_dist())  # cost of all features in a cluster
             
            cost_table[label_clust] = (np.min(cost_list))  
            
        match = min(cost_table, key = cost_table.get)
        cost = sorted(cost_table.items(),key=itemgetter(1),reverse=False)[0]
        print('label for ',file_id[0],match)
        
        
        if match not in results_cluster:
            
            results_cluster[match]= [(file_id,cost[1])]
                        
        
        else:
             results_cluster[match].append((file_id,cost[1]))
    
    file = open('/results.txt', "w+")
    with open("/task/keywords_test.txt", "r") as myfile:
        
        lines = myfile.readlines()
        
        for line in lines:
            line = line.replace("\n","")
            keyword, imageid = line.split(",")
            print('test', keyword)
            stng = keyword
            for i in results_cluster:
                if i == keyword:
                    for j in results_cluster[i]:
                    #print(j)
                        r_id = j[0][0].split('\\')[1]
                        cost = j[1]
                        
                        stng = stng +' '+ r_id  +','+str(cost)+' '
                        
                    file.write(stng)
    file.close()




if __name__ == "__main__":
    main()

import similarclusters as sc
import getdata
import glob
from sklearn import preprocessing
from cdtw import pydtw
import numpy as np

def main():
    print("Validation process starting...")
    #sc.loadFeatureVector_tofile()
    print("All features loaded...")

    clusters = sc.initialize_image_clusters()
    with open("/task/keywords_test.txt", "r") as myfile:
        min_max_scaler = preprocessing.MinMaxScaler()
        lines = myfile.readlines()
    
        for line in lines:
            line = line.replace("\n","")
            keyword, imageid = line.split(",")
            print('test', keyword)
    
            # For each images to classify and have the dissimilarity
            for filename in glob.glob('/valid/*'):
                file_id = filename.split(".png")
                print('testing ', file_id)
                
    
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
                    m = np.min(cost_list)
                    cost_table[label_clust] = (m)  
                    # seems to work sometimes,len([num for num in cost_list if num  < m+s]) )
                    #print('top 4 ', sorted(cost_table.items(),key=itemgetter(1),reverse=False)[0:4])
                    match = min(cost_table, key = cost_table.get)
                    
                    print('label for ',file_id,min(cost_table, key = cost_table.get))

                # Print a line for each keywords (10 lines - 10 clusters) with all matches to each local images (305-xxx)
                # furnished by the UNIFR as testing set
                list_of_cost = [keyword]
                for cost in cost_table:
                    list_of_cost.append(cost[0])
                    list_of_cost.append(cost[1])




if __name__ == "__main__":
    main()

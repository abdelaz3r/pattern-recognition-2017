
# coding: utf-8

# University of Fribourg
# 
# Departement of Informatics
# 
# SS 2017
# 
# *********************
# # Pattern Recognition
# 
# ## Important to test the algorithm you just have to put the test and train csv files in the same file, and run the code!
# 
# n.b.: Best results with k = 5 or 3, around 97 %
# 

# In[11]:


import time
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


# In[12]:

# Make rows names more readable
def new_columns_names(columns_size):
    new_columns_array = np.array(['label'])

    for i in range(1,size):
        new_name = 'pixel'+ str(i)
        new_columns_array = np.append(new_columns_array, [new_name])
        
    return new_columns_array

# Read csv files
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

#Create the corresponding data frames
train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)

#Size of columns array must be the same between training data and test data
size = len(train_df.columns.values)

#Compute the new data frame
train_df.columns = new_columns_names(size)
test_df.columns =  new_columns_names(size)


# In[13]:

#KNN is non-parametric, instance-based and used in a supervised learning setting.
#X_train reprensents features of our taining set --> pixels
#Y_train represents labels of our training set --> number in the image

len(train_df.ix[0].values)
train_df.values[0]
#select pixels row, put all values an array
X_train = train_df.filter(regex = ("pixel.*")).values
X_test = test_df.filter(regex = ("pixel.*")).values

y_train = train_df['label'].values
y_test = test_df['label'].values

#drop to increase time for testing the distances method accuracy
#X_test = np.delete(X_test,np.s_[100::],0)
#y_test = np.delete(y_test, np.s_[100::])

#print the sahpe
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape )


# ##  Core of the algorithm
# 
# * Training block
# * Distances definition
# * Predict block
# * Define the kNN
# 

# In[14]:

#Training block of the kNN, nothing to do: instance based algorithm
def train(X_train, y_train):
    #do nthing
    return


# In[15]:

#Define two kind of distances
def manhattan_distance(X_train, X_test):
    #create list for distances
    distances = []
    
    for i in range(len(X_train)):
        #compute the distance
        distance = np.sum(np.abs(X_test- X_train[i,:]))
        #add it to list of distances
        distances.append([distance,i])
        
    #sort the list
    distances = sorted(distances)
    
    return distances

def euclidian_distance(X_train, X_test):
    #create list for distance
    distances = []
    
    for i in range(len(X_train)):
        #compute the distance
        distance = np.sqrt(np.sum(np.square(X_test - X_train[i,:])))
        #add it to list of distances
        distances.append([distance,i])
        
    #sort the list
    distances = sorted(distances)
    
    return distances
    

#Predict block with manhattan distance
def predict(X_train, y_train, X_test, k):
    #create list for distances and labels
    distances = euclidian_distance(X_train, X_test)
    labels = []
    #make a list of the k neighbors'targets
    for i in range(k):
        index = distances[i][1]
        labels.append(y_train[index])
    
    #return most common label
    return Counter(labels).most_common(1)[0][0]


# In[16]:

#KNN
def kNN(X_train,y_train,X_test, predictions, k):
        #train on the input data
        train(X_train, y_train)
        #loop over all observations
        for i in range(len(X_test)):
            predictions.append(predict(X_train, y_train, X_test[i, :], k))


# ## Time issues
# 
# One way to cut down the curse of dimensionality of our set is to try to decompose the data and restructure it using some technics like
# 
# * KD-tree
# * Ball tree
# * Principal component analysis
# 
# We choos the PCA algorithm already implementend in the sklearn library
# 
# * Separate the feature space in visible cluster for 2 components
# * Try to capture the most of the variance in the dataset predicting how the prediction is good regarding the number of components
# * Choose a #of components avoiding overfitting, regarding the function
# * Compute the transform sets with the kNN algorithm increasing the speed of the algorithm

# In[ ]:

pca = PCA(n_components=2)
pca.fit(X_train)
transform = pca.transform(X_train)



# In[ ]:

n_components_array=([1,2,3,4,5,10,20,50,100,200,500])
vr = np.zeros(len(n_components_array))
i=0;
for n_components in n_components_array:
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    vr[i] = sum(pca.explained_variance_ratio_)
    i=i+1  


# In[ ]:

pca = PCA(n_components = 50)
pca.fit(X_train)
transform_train = pca.transform(X_train)
transform_test = pca.transform(X_test)


# ## Run the algorithm with the transform train and test sets

# In[ ]:

#Run the algorithm
predictions = []

#there divide in 10 tasks
tic = time.time()
kNN(transform_train, y_train, transform_test, predictions, 5)
toc = time.time()
print(toc-tic)

#transform the list into an array
predictions = np.asarray(predictions)

#accuracy 
accuracy = accuracy_score(y_test, predictions)
print('\nThe accuracy of our classifier is %d%%' % (accuracy*100))


# In[ ]:

#Save the output file
out_file= open("predictionsK5Euc.txt", "w")
out_file.write("ImageID, Label \n")
for i in range(len(predictions)):
    out_file.write(str(i+1) + "," + str(int(predictions[i])) + "\n")
out_file.close()


# In[ ]:




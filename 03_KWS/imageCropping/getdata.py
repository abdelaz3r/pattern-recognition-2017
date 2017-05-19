
import numpy as np
from PIL import Image



def getfeatures(filename):
    img = (Image.open(filename).resize((200, 200)))
    img = np.asarray(img)
    
    feature_matrix = np.zeros((200,7))
    for i in range(200):
        #Get all feature for each column
        column = img[:,i] # single pixel column(sliding window)
        udx,ldx = get_countour(column,img,i)
        feature_matrix[i,0] = udx
        feature_matrix[i,1] = ldx
        feature_matrix[i,2],feature_matrix[i,3] = blk2wht(column)
        feature_matrix[i,4] = get_no_win_blk(column)
        
        if(ldx > 0 and udx > 0):
            feature_matrix[i,5] = get_ratio_blkinCountour(column,udx,ldx)
        else:
            feature_matrix[i,5] = 0
        
        feature_matrix[i,6] = get_countour_grad_change(img,i,ldx,udx,column)
        
        
    return feature_matrix
       

def get_countour(column,img,i):
    print('column',i)
    #1upper countour ? values or index
    uc = np.where(column == 0)[0] #first instance at column 0
    if uc.size > 0:
        udx = uc[0]# index
    else :
        udx = 0
    
    #2 lower countour, reverse column, first black instance
    lc = np.where(np.flipud(img[:,i] == 0))[0]
    if lc.size > 0 :
        ldx = 200 - lc[0]# index
    else :
        ldx = 0
    return udx,ldx


def blk2wht(column):       
    #  get counts of balck and white, unique number and their counts
    unique, counts = np.unique(column, return_counts=True)
    bwn =dict(zip(unique, counts))
    print (bwn)
    
    #3get black to white ratio
    if(bwn.get(0) and bwn.get(255)):
        bw_ratio= bwn.get(0)/bwn.get(255)# if black and white are 0's and 1's
    else :
        bw_ratio = 0
     # 4 black to white transitions   
    counter = 0
    for j in range(1,200):
        if column[j] != column[j-1]:
            counter = counter + 1
    return bw_ratio, counter   
        
   
def get_no_win_blk(column):        
            
    #5 number of black in window
    counter = 0
    for j in range(0, len(column)):
        if column[j] == 0:
            counter = counter +1
    if counter:
        b_ratio = counter/len(column)
    
    else:
        b_ratio = 0
    return b_ratio
    
 
def get_ratio_blkinCountour(column, ldx, udx):
    # 6 Black pixels fraction between LC and UC
    counter = 0
    for j in range(udx,ldx ):
        if column[j] == 0:
            counter = counter +1
    if counter :
        blackrange =  counter/len(range(udx,ldx))
    else:
        blackrange = 0
    return blackrange
                 
def get_countour_grad_change(img,i,ldx,udx,column): # if the not the last column
    if i < 199:
        column_plus = img[:,i+1]
        uc_plus = np.where(column_plus == 0)[0] #first instance at column 0
        if  uc_plus.size > 0: #if no upper countour space is white
            udx_plus = uc_plus[0]# index        
            # lower countour, reverse column, first black instance
            lc_plus = np.where(np.flipud(img[:,i+1] == 0))[0]
            ldx_plus = 200 - lc_plus[0]# index
        
            # 7 gradient change
            b_grad = abs(ldx - udx) - (ldx_plus - udx_plus)
        else :
            b_grad = 0
        return b_grad
    else:
        return 0
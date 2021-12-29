#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Do All permutations
import itertools
List_permutations=["".join(Perm) for Perm in itertools.product("01", repeat=12)]
List_permutations[5]


# In[2]:


resolution=5

# Import Complexity metric

import gc
gc.collect()
import numpy as np
import random
from numpy.random import seed


# In[3]:


res=resolution
#Generate tensor that represents the wedges 
##1) Generate the wedges as tensors 
import numpy as np

wedge1=np.ones((res, res,res), dtype = int)
wedge1=np.tril(wedge1)

import numpy as np


wedge2=np.flip(wedge1, axis=1)
wedge3=np.flip(wedge1, axis=(1,2))
wedge4=np.flip(wedge1, axis=(2))
wedge5=np.rot90(wedge1, k=1, axes=(0, 1))
wedge6=np.flip(wedge5, axis=(2))
wedge7=np.flip(wedge5, axis=(2,0))
wedge8=np.flip(wedge5, axis=(1,0))
wedge9=np.rot90(wedge1, k=1, axes=(0, 2))
wedge10=np.flip(wedge9, axis=(1))
wedge11=np.flip(wedge9, axis=(1,0))
wedge12=np.flip(wedge9, axis=(2,0,))


# In[4]:


#This is the vector of length 12 that control which wedges to have.

#Now we need to merge multiple wedhges and output a new set of 6 images
def merge_wedges(action):
  # We muptipli the action vector for each of the wedges and them add them all together
    wedge_1t=np.multiply(wedge1,action[0])
    wedge_2t=np.multiply(wedge2,action[1])
    wedge_3t=np.multiply(wedge3,action[2])
    wedge_4t=np.multiply(wedge4,action[3])
    wedge_5t=np.multiply(wedge5,action[4])
    wedge_6t=np.multiply(wedge6,action[5])
    wedge_7t=np.multiply(wedge7,action[6])
    wedge_8t=np.multiply(wedge8,action[7])
    wedge_9t=np.multiply(wedge9,action[8])
    wedge_10t=np.multiply(wedge10,action[9])
    wedge_11t=np.multiply(wedge11,action[10])
    wedge_12t=np.multiply(wedge12,action[11])
    #add all wedges
    shape=wedge_1t++wedge_2t++wedge_3t++wedge_4t++wedge_5t++wedge_6t++wedge_7t++wedge_8t++wedge_9t++wedge_10t++wedge_11t++wedge_12t
    shape = np.where(shape > 1, 1, shape)
    return shape


# In[5]:
#For eachof the 3 axis planes do"
#1)move the 2D plane on the given axis , only consider  non emtry plane
#2) For each nonemtpy plane, reduce the place to a rectangle or sq that has 1s
#3) For each reduce nonemtpy plane chehek for lines of symetri differences (Vertical, Horisontal, 2 45Degress)

def simetric_of_plane(final_shape,plane):
    simetry_dis=0
    for p in range(1): # final_shape.shape[0]):
        
        if(plane=="z"):
            cutting_plane=final_shape[:,:,p]
        if(plane=="y"):
            cutting_plane=final_shape[:,p,:]
        if(plane=="x"):
            cutting_plane=final_shape[p,:,:]
            
        if p!=0:
            if np.linalg.norm(final_shape[p,:,:]-final_shape[p-1,:,:])==0:
                continue    
            
        if(sum(sum(cutting_plane))>0):   #if the plane is not empty
            
            #Get index of to reduce plane
            indexs=np.argwhere(cutting_plane==1)
            h_index_max=max(indexs[:,0])+1
            h_index_min=min(indexs[:,0])
            v_index_max=max(indexs[:,1])+1
            v_index_min=min(indexs[:,1])

            reduce_plane=cutting_plane[h_index_min:h_index_max, v_index_min:v_index_max]

            #vertical simetry diff
            v_reduce=reduce_plane
            if(not (v_reduce.shape[1]%2==0)):
                v_reduce=np.delete(v_reduce,round(v_reduce.shape[1]/2)-1,1)
            sub_arrays=np.split(v_reduce,2,axis=1)

            vertical_sim_dis=np.linalg.norm(sub_arrays[1]-sub_arrays[0])/((sub_arrays[1].shape[0]+sub_arrays[1].shape[1])/2)

            #horizontal simetry diff
            h_reduce=reduce_plane
            if(not (h_reduce.shape[0]%2==0)):
                h_reduce=np.delete(h_reduce,round(h_reduce.shape[0]/2)-1,0)
            sub_arrays=np.split(h_reduce,2,axis=0)

            horizontal_sim_dis=np.linalg.norm(sub_arrays[1]-sub_arrays[0])/((sub_arrays[1].shape[0]+sub_arrays[1].shape[1])/2)

            #diagonal 1 simetry diff
            lower_tril=np.tril(reduce_plane)
            upper_tril_flip=np.rot90(np.fliplr(np.triu((reduce_plane))))
            #Check of diagonal matrix are the same size, otherwise add 0
            if(lower_tril.shape[0]>upper_tril_flip.shape[0]):
                rows_to_add=lower_tril.shape[0]-upper_tril_flip.shape[0]
                z = np.zeros((rows_to_add,upper_tril_flip.shape[1]))
                upper_tril_flip=np.append(upper_tril_flip, z, axis=0)
            if(lower_tril.shape[0]<upper_tril_flip.shape[0]):
                rows_to_add=upper_tril_flip.shape[0]-lower_tril.shape[0]
                z = np.zeros((rows_to_add,lower_tril.shape[1]))
                lower_tril=np.append(lower_tril, z, axis=0)

            if(lower_tril.shape[1]>upper_tril_flip.shape[1]):
                cols_to_add=lower_tril.shape[1]-upper_tril_flip.shape[1]
                z = np.zeros((upper_tril_flip.shape[0],cols_to_add))
                upper_tril_flip=np.append(upper_tril_flip, z, axis=1)
            if(lower_tril.shape[1]<upper_tril_flip.shape[1]):
                cols_to_add=upper_tril_flip.shape[1]-lower_tril.shape[1]
                z = np.zeros((lower_tril.shape[0],cols_to_add))
                lower_tril=np.append(lower_tril, z, axis=1)

            diag1_sim_dis=np.linalg.norm(lower_tril-upper_tril_flip)/((lower_tril.shape[0]+lower_tril.shape[1])/2)


            #diagonal 2 simetry diff
            lower_tril=np.tril(np.fliplr(reduce_plane))
            upper_tril_flip=np.rot90(np.fliplr(np.triu(np.fliplr(reduce_plane))))
                    #Check of diagonal matrix are the same size, otherwise add 0
            if(lower_tril.shape[0]>upper_tril_flip.shape[0]):
                rows_to_add=lower_tril.shape[0]-upper_tril_flip.shape[0]
                z = np.zeros((rows_to_add,upper_tril_flip.shape[1]))
                upper_tril_flip=np.append(upper_tril_flip, z, axis=0)
            if(lower_tril.shape[0]<upper_tril_flip.shape[0]):
                rows_to_add=upper_tril_flip.shape[0]-lower_tril.shape[0]
                z = np.zeros((rows_to_add,lower_tril.shape[1]))
                lower_tril=np.append(lower_tril, z, axis=0)

            if(lower_tril.shape[1]>upper_tril_flip.shape[1]):
                cols_to_add=lower_tril.shape[1]-upper_tril_flip.shape[1]
                z = np.zeros((upper_tril_flip.shape[0],cols_to_add))
                upper_tril_flip=np.append(upper_tril_flip, z, axis=1)
            if(lower_tril.shape[1]<upper_tril_flip.shape[1]):
                cols_to_add=upper_tril_flip.shape[1]-lower_tril.shape[1]
                z = np.zeros((lower_tril.shape[0],cols_to_add))
                lower_tril=np.append(lower_tril, z, axis=1)

            diag2_sim_dis=np.linalg.norm(lower_tril-upper_tril_flip)/((lower_tril.shape[0]+lower_tril.shape[1])/2)
            simetry_dis=(simetry_dis+(vertical_sim_dis+horizontal_sim_dis+diag1_sim_dis+diag2_sim_dis)/4)/2
        

    gc.collect()
    return simetry_dis

#Calculate Complexity of the new shape using euclidian distance at pixellevel
#This complexity is calculated using also the inner "pixels"




def complexity_cal(final_shape):
    complexity=(simetric_of_plane(final_shape,'x')+simetric_of_plane(final_shape,'y')+simetric_of_plane(final_shape,'z'))/3
    gc.collect()
    return np.round(complexity,2)


# In[3]:


#Test complecity

action_v=np.array(list(List_permutations[120]),dtype=int)
final_shape=merge_wedges(action_v)
print(complexity_cal(final_shape)) 



# In[15]:


new_list = []
new_complexitylist = []

from tqdm.notebook import trange, tqdm
for i in tqdm(range(len(List_permutations))):
    action_v=np.array(list(List_permutations[i]),dtype=int)
    final_shape=merge_wedges(action_v)
    a=complexity_cal(final_shape)
    
    new_complexitylist.append(a)
    if a == 0:
        new_list.append(i)
        
import pandas as pd  
new_complexitylist=pd.DataFrame(new_complexitylist)
new_complexitylist.to_csv('C:/Users/Asus/OneDrive/Desktop/Project/complexitylist1.csv')

new_list=pd.DataFrame(new_list)
new_list.to_csv('C:/Users/Asus/OneDrive/Desktop/Project/indexlist1.csv')


# In[4]:


i = 0
while i < len(List_permutations):
   action_v=np.array(list(List_permutations[i]),dtype=int)
   final_shape=merge_wedges(action_v)
   a=complexity_cal(final_shape)
   i += 1


# In[ ]:



colors = ["red", "green", "blue", "purple"]
for i in range(len(colors)):
    print(colors[i])


# In[6]:


get_ipython().system('pip install tqdm')


# In[11]:


print(new_complexitylist)


# In[12]:


import pandas as pd  
new_complexitylist=pd.DataFrame(new_complexitylist)
new_complexitylist.to_csv('C:/Users/Asus/OneDrive/Desktop/Project/complexitylist.csv')


# In[13]:


new_list=pd.DataFrame(new_list)
new_list.to_csv('C:/Users/Asus/OneDrive/Desktop/Project/indexlist.csv')


# In[ ]:





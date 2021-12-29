#!/usr/bin/env python
# coding: utf-8

# In[2]:


#INPUT: Resolution of images
import time
ts = time.time()
res=5


# In[3]:


#Generate tensor that represents the wedges 
##1) Generate the wedges as tensors 
import numpy as np

wedge1=np.ones((res, res,res), dtype = int)
wedge1=np.tril(wedge1)
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

import ipyvolume as ipv
ipv.quickvolshow(wedge1,level=[1, 1], opacity=1, data_min=0, data_max=1)


# In[4]:


#This is the vector of length 12 that control which wedges to have.

#Now we need to merge multiple wedhges and output a new set of 6 images
def merge_wedges_single_voxel(action):

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
  voxel=wedge_1t++wedge_2t++wedge_3t++wedge_4t++wedge_5t++wedge_6t++wedge_7t++wedge_8t++wedge_9t++wedge_10t++wedge_11t++wedge_12t

  voxel = np.where(voxel > 1, 1, voxel)
  return voxel


# In[5]:


#Calculate Complexity of the new shape using euclidian distance at pixellevel
#This complexity is calculated using also the inner "pixels"
def complexity_cal(new_shape):
    complexity=0
    
    for p in range(new_shape.shape[0]):
        for p_in  in range(0,new_shape.shape[0]):
                
                complexity=complexity+np.linalg.norm(new_shape[:,:,p] - new_shape[:,:,p_in])
                complexity=complexity+np.linalg.norm(new_shape[:,p,:] - new_shape[:,p_in,:])
                complexity=complexity+np.linalg.norm(new_shape[p,:,:] - new_shape[p_in,:,:])
    complexity=complexity/(6*new_shape.shape[0])
    return complexity


# In[6]:


#This functions merges all the final voxels into one final shape
def merge_voxels(action_v, over_p):

    final_shape=np.zeros(((res*2), (res*2),(res*2)), dtype = int)

    #Just if we need  to overlap to plot
    over=over_p
    final_shape[0:res,0:res,0:res]=final_shape[0:res,0:res,0:res]++merge_wedges_single_voxel(action_v[0])
    final_shape[(res-over):((res*2)-over),0:res,0:res]=final_shape[(res-over):((res*2)-over),0:res,0:res]++merge_wedges_single_voxel(action_v[1])
    final_shape[(res-over):((res*2)-over),(res-over):((res*2)-over),0:res]=final_shape[(res-over):((res*2)-over),(res-over):((res*2)-over),0:res]++merge_wedges_single_voxel(action_v[2])
    final_shape[(res-over):((res*2)-over),(res-over):((res*2)-over),(res-over):((res*2)-over)]=final_shape[(res-over):((res*2)-over),(res-over):((res*2)-over),(res-over):((res*2)-over)]++merge_wedges_single_voxel(action_v[3])

    final_shape[0:res,(res-over):((res*2)-over),0:res]=final_shape[0:res,(res-over):((res*2)-over),0:res]++merge_wedges_single_voxel(action_v[4])
    final_shape[0:res,(res-over):((res*2)-over),(res-over):((res*2)-over)]=final_shape[0:res,(res-over):((res*2)-over),(res-over):((res*2)-over)]++merge_wedges_single_voxel(action_v[5])

    final_shape[(res-over):((res*2)-over),0:res,(res-over):((res*2)-over)]=final_shape[(res-over):((res*2)-over),0:res,(res-over):((res*2)-over)]++merge_wedges_single_voxel(action_v[6])
    final_shape[0:res,0:res,(res-over):((res*2)-over)]=final_shape[0:res,0:res,(res-over):((res*2)-over)]++merge_wedges_single_voxel(action_v[7])

    return final_shape


# In[27]:


action_v=[[1,0,0,0,1,0,0,0,1,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,1,0,0,0,1,1,0,0,0,0]]

nums=[1,2,3]

final_shape=merge_voxels(action_v,0)
print(complexity_cal(final_shape))
ipv.quickvolshow(final_shape,level=[1, 1], opacity=1,data_min=0, data_max=1) 


product (range(2), repeat=12)
print (product(range(2),repeat=12))


# In[26]:


action_v=[[1,1,1,1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [1,0,0,0,0,0,0,0,0,0,0,0],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [1,0,0,0,0,0,0,0,0,0,0,0],
          [0,1,0,0,0,0,0,0,0,0,0,0],
          [0,1,0,0,0,0,0,0,0,0,0,0]]

final_shape=merge_voxels(action_v,0)
print(complexity_cal(final_shape))
ipv.quickvolshow(final_shape,level=[1, 1], opacity=1,data_min=0, data_max=1)


# In[23]:


action_v=[[1,1,1,1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0]]

final_shape=merge_voxels(action_v,0)
print(complexity_cal(final_shape))
ipv.quickvolshow(final_shape,level=[1, 1], opacity=1,data_min=0, data_max=1)


# In[45]:


action_v=[[1,1,1,1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,1,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0]]

final_shape=merge_voxels(action_v,0)
print(complexity_cal(final_shape))
ipv.quickvolshow(final_shape,level=[1, 1], opacity=1,data_min=0, data_max=1)


# In[24]:


action_v=[[1,1,1,1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [1,1,1,1,1,1,1,1,1,1,1,1]]

final_shape=merge_voxels(action_v,0)
print(complexity_cal(final_shape))
ipv.quickvolshow(final_shape,level=[1, 1], opacity=1,data_min=0, data_max=1)


# In[29]:


action_v=[[0,0,0,0,0,0,0,1,0,0,0,0],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [0,1,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0]]

final_shape=merge_voxels(action_v,0)
print(complexity_cal(final_shape))
ipv.quickvolshow(final_shape,level=[1, 1], opacity=1,data_min=0, data_max=1)


# In[21]:


action_v=[[1,1,1,1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,1,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [1,1,1,1,1,1,1,1,1,1,1,1]]

final_shape=merge_voxels(action_v,0)
print(complexity_cal(final_shape))
ipv.quickvolshow(final_shape,level=[1, 1], opacity=1,data_min=0, data_max=1)


# In[14]:


action_v=[[1,1,1,1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [1,1,1,1,1,1,1,1,1,1,1,1],
          [0,0,0,0,0,0,0,1,0,0,0,0]]

final_shape=merge_voxels(action_v,0)
print(complexity_cal(final_shape))
ipv.quickvolshow(final_shape,level=[1, 1], opacity=1,data_min=0, data_max=1)


# In[23]:


te = time.time()
print(te-ts)


# In[17]:


action_v=[[1,0,0,0,1,0,0,0,1,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0]]



final_shape=merge_voxels(action_v,0)
print(complexity_cal(final_shape))
ipv.quickvolshow(final_shape,level=[1, 1], opacity=1,data_min=0, data_max=1) 


# In[18]:


action_v=[[1,1,1,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0]]



final_shape=merge_voxels(action_v,0)
print(complexity_cal(final_shape))
ipv.quickvolshow(final_shape,level=[1, 1], opacity=1,data_min=0, data_max=1) 


# In[22]:


action_v=[[0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0]]



final_shape=merge_voxels(action_v,0)
print(complexity_cal(final_shape))
ipv.quickvolshow(final_shape,level=[1, 1], opacity=1,data_min=0, data_max=1) 


# In[ ]:





# In[ ]:

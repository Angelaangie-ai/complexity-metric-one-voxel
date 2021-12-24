#!/usr/bin/env python
# coding: utf-8

# In[2]:


#INPUT: Resolution of images
res=20


# In[3]:


#Generate tensor that represents the wedges 
##1) Generate the wedges as tensors 
import numpy as np

wedge1=np.ones((res, res,res), dtype = int)
wedge1=np.tril(wedge1)

import numpy as np
import ipyvolume as ipv
ipv.quickvolshow(wedge1,level=[1, 1], opacity=1, data_min=0, data_max=1)


# In[4]:


wedge2=np.flip(wedge1, axis=1)
ipv.quickvolshow(wedge2,level=[1, 1], opacity=1, data_min=0, data_max=1)


# In[5]:


wedge3=np.flip(wedge1, axis=(1,2))
ipv.quickvolshow(wedge3,level=[1, 1], opacity=1, data_min=0, data_max=1)


# In[5]:


wedge4=np.flip(wedge1, axis=(2))
#ipv.quickvolshow(wedge4,level=[1, 1], opacity=1, data_min=0, data_max=1)


# In[6]:


wedge5=np.rot90(wedge1, k=1, axes=(0, 1))
ipv.quickvolshow(wedge5,level=[1, 1], opacity=1)


# In[7]:


wedge6=np.flip(wedge5, axis=(2))
#ipv.quickvolshow(wedge6,level=[1, 1], opacity=1)


# In[8]:


wedge7=np.flip(wedge5, axis=(2,0))
#ipv.quickvolshow(wedge7,level=[1, 1], opacity=1)


# In[9]:


wedge8=np.flip(wedge5, axis=(1,0))
#ipv.quickvolshow(wedge8,level=[1, 1], opacity=1)


# In[10]:


wedge9=np.rot90(wedge1, k=1, axes=(0, 2))
ipv.quickvolshow(wedge9,level=[1, 1], opacity=1)


# In[11]:


wedge10=np.flip(wedge9, axis=(1))
#ipv.quickvolshow(wedge10,level=[1, 1], opacity=1)


# In[12]:


wedge11=np.flip(wedge9, axis=(1,0))
#ipv.quickvolshow(wedge11,level=[1, 1], opacity=1)


# In[13]:


wedge12=np.flip(wedge9, axis=(2,0,))
#ipv.quickvolshow(wedge12,level=[1, 1], opacity=1)


# In[14]:


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


# In[1]:


#Calculate Complexity of the new shape using euclidian distance at pixellevel
#This complexity is calculated using also the inner "pixels"


def complecity_cal(new_shape):
    complexity=0
    
    for p in range(new_shape.shape[0]):
        for p_in  in range(0,new_shape.shape[0]):
                
                complexity=complexity+np.linalg.norm(new_shape[:,:,p] - new_shape[:,:,p_in])
                complexity=complexity+np.linalg.norm(new_shape[:,p,:] - new_shape[:,p_in,:])
                complexity=complexity+np.linalg.norm(new_shape[p,:,:] - new_shape[p_in,:,:])
    complexity=complexity/(6*new_shape.shape[0])
    return complexity


# In[24]:


action_v=[1,0,0,0,0,0,0,0,0,0,1,0] 
shape=merge_wedges(action_v)
print(complecity_cal(shape))
ipv.quickvolshow(shape,level=[1, 1], opacity=1,data_min=0, data_max=1) 


# 
# 

# In[25]:


points=np.transpose((shape>0).nonzero())
points


# In[26]:


import pyvista as pv
point_cloud = pv.PolyData(points)
point_cloud


# In[42]:


volume = point_cloud.delaunay_3d(alpha=0.868)
shell = volume.extract_geometry()
shell.plot(lighting=True)


# In[37]:


#!pip install ipygany
plotter = pv.Plotter(notebook=True)
plotter.add_mesh(volume)
plotter.show(jupyter_backend='ipygany')


# In[21]:


import pyvista as pv
sphere = pv.Sphere()
plotter = pv.Plotter(notebook=True)
plotter.add_mesh(sphere)
plotter.show(jupyter_backend='ipygany')


# In[22]:


point_cloud


# In[39]:


sphere


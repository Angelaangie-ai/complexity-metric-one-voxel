# Spatial Visualization Skills

Spatial visualization skills are critical to further develop and advance in the fields of STEM. While most students have underdeveloped visualization skills, there are 
educational tools that can help them develop such skills, like Virtual Reality Applications.  But in order to develop to keep students engaged, we need that present new 
challenges and content suitable. In order to train a model that is capable of generating shapes, we first need a uniform system capable of measuring their complexity.

To model the complexity of the 3D Shapes, we’ve collected the main data with the help of Amazon Mechanical Turk. AMT is a crowdsourcing platform where individuals get 
paid to perform human intelligence tasks. Through AMT we’ve analyzed three data sets 360 degrees rotating videos, 3D Shapes images, and Questions from the 
Purdue Visualization Test. 

- The official videos used for research purposes are present here: https://drive.google.com/drive/folders/1GcoPM7yxnD7zlwxdBEVzAefYQHuaORF-?usp=sharing
_________________________________________________________________________________________________________________________________________________________________________________
We’ve compared the results with our complexity-measuring algorithm, which is based on calculating the symmetry of 3D shapes using their symmetry lines. 
The results indicate that there is a significant  correlation between the participants' perceived shape complexity and our complexity metric.  

- One voxel complexity: The first one generated a tensor that represented the wedges and further generated the wedges as tensors. We’ve added the vector of length 12 
(the 3D shape) which controlled which wedges are present in the body. The decision on which tensors are present in the body decides the shape. We’ve multiplied the 
action vector for each of the wedges and added them all together. On the given body, we’ve calculated the complexity of the new shape using euclidian distance at the pixel level.
- Eight voxel complexity: The second version as well generated a tensor that represented the wedges and further generated them as tensors. This is the vector of length 12 that control which wedges to have. Now we need to merge multiple wedges and output a new set of 6 images. In the same way, we calculate the complexity of the new shape using euclidian distance at the pixel level.
__________________________________________________________________________________________________________________________________________________________________________________
After presenting our poster, at the conference, we managed to reach out to numerous individuals and ask what are the determining properties people use to define whether a shape is complex or not.

The reasons people perceive include but are not limited to:
Amount of building blocks that we are able to see on the shape
The number of cubes, pyramids, etc. of which the shape is produced

Here‘s the poster: https://www.canva.com/design/DAEs8bD77yQ/VutpbP9w6_YbygVV_cVsow/view?utm_content=DAEs8bD77yQ&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink  


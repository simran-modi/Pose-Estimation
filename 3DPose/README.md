
## Introduction
Current approaches cannot scale to large-scale problems because they rely on one classifier per object, or multi-class classifiers such as Random Forests, whose complexity grows with the number of objects.
**The only recognition approaches that have been demonstrated to work on large scale problems are based on *Nearest Neighbor (NN)* classification **

While  **feature  point  descriptors  are  used  only  to  find  the points'  identities,  we  here  want  to  find  both  the  object's
identity and its pose**.
We therefore seek to **learn a descriptor with the two following properties**:
- The Euclidean distance between descriptors from two different objects should be  large;
- The  Euclidean  distance  between  descriptors from the same object should be representative of the similarity between their poses.
This way, given a new object view, we can **recognize the object and get an estimate of its pose by matching its descriptor against a database of registered descriptors.**

## Method
**CNN is used to identify descriptors which are stored during training in database.
While testing, the CNN is used to compute descriptors which is compared to the database to find the nearest neighbour**
#####Working
For each object in the database,** descriptors are calculated for a set of template views and stored along with the object's identity and 3D pose of the view.** In order to get an estimate for the class and pose of the object depicted in the new input image, we can **compute a descriptor for x** and **search for the most similar descriptors in the database.** The output is then the object and pose associated with them. Therefore, we introduce a method to efficiently map an input image to a compact and discriminative descriptor that can be used in the nearest neighbor search according to the Euclidean distance.
#####Training the CNN
*S~train~ training samples*
Each sample = (x,c,p)
- **x**, **image** of an object, (color or grayscale)
- **c**, the **identity** of the object
- **p**, the **3D pose** of the object relative to the camera

*S~db~ templates *
Each element is defined in the same way as a training sample.
Descriptors for these templates are calculated and stored with the classifier for k-nearest neighbor search.
The template set can be a **subset of the training set, the whole training set or a separate set.**

##  Defining the Cost Function
***L=L ~triplets~ + L ~pairs~ + Lambda||w'||~2~^2^***
**w′ -** vector made of all the weights of the convolutional filters
##### L ~triplets~
Create a triplet of training samples - (s~i~,s~j~, s~k~)
Such that, either
* s~i~ and s~j~ are from the same object and s~k~ from a different object
* p~i~ and p~j~ are similar poses but p~i~ and p~k~ are not

![ ](https://www.evernote.com/shard/s666/res/38246681-d6d0-4991-9695-3876f5ccd7df/tripletsmin.png  "Cost Formula")
**The margin m **
* introduces a **margin** for the classification
* defines a **minimum ratio for the Euclidean distances of the dissimilar pair of samples and the similar** one.
*default value - 0.01*
##### L ~pairs~
Use pair-wise terms. These terms **make the descriptor robust to noise** and other distracting artifacts such as changing illumination.
Consider a pair (s~i~,s~j~) of samples from the **same object** under **very similar poses**, ideally the same, and we
This term therefore **enforces the fact that for two images of the same object and same pose, we want to obtain two descriptors which are as close as possible to each other**, even if they are from different imaging conditions:
![](/home/b/Pictures/pairsmin.png)

##Implementation

* two **convolutional layers** with a set of filters,max-pooling and sub-sampling over a 2x2 area and a rectified  linear  (ReLU)  activation  function
* two **fully connected layers**.
	* The first fully connected layer also employs a ReLU activation
	* the last layer has linear output and delivers the final descriptor.

Parameters optimised by Stochaistic Gradient Descent.
To assemble a mini-batch

* we start by randomly taking one training sample from each object
* for each of them we add its template with the most similar pose, unless it was already included in this mini-batch.
* However, this procedure can lead to very unequal numbers of templates per object if, for instance, all of the selected training samples have the same most similar template.
* We make sure that for each object at least two templates are included by adding a random one if necessary.
* For each training sample in the mini-batch,
	* we initially create three triplets
	* In each of them the similar template is set to be the one with the closest pose
	* Dissimilar sample is either another, less similar template from the same object or any template of a different object.
* **Pairs are then formed by associating each training sample with its closest template**

##Dataset Compilation
We train a CNN using our method on a mixture of ***synthetic*** and ***real world data***.  We create synthetic training data by **rendering the mesh available for each of the objects in the dataset from positions on a half-dome over the object**.
Viewpoints are defined by *starting with a regular icosahedron* and *recursively subdividing each triangle into 4 sub-triangles*.
Hence, totally - 1241 positions.

(Alternatively using matplotlib to take snaps of the the picture by varying Azimuth and elevation)
(Additionally consider rotational invariance of objects too and not just azimuth/elevation)

For the real world data, we **split the provided sequences captured with the Kinect randomly into a training and a test set**.   We  ensure  an even  distribution  of  the  samples  over the  viewing  hemisphere  by  taking  **two  real  world  images close to each template**, which **results roughly in a 50/50 split** of the data into training and test.
x
###### Adding Noise
**Make multiple copies of training data with added noise. **
On both RGB and depth channel we add a small amount of Gaussian noise.
On the synthetic images, we add larger fractal noise on the background, **to simulate diverse backgrounds**

######Normalisation
* RGB images - normalized to the usual zero mean, unit variance.
* Depth maps - subtract the depth at the center of the object, scale down such that 20 cm in front and back map to -1 to +1

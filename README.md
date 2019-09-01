# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Datasets for training, validation and test:
-train.p
-valid.p
-test.p

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630s
* The shape of a traffic sign image is: 32x32
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.
-Visualization of a random sign from the dataset:
[image6]: ./Figures/Sign.png

-Visualization of distribution of the classes in training(blue), validation(red) and test(green) datasets:
[image7]: ./Figures/Distribution.png

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Only normalization is used as a preprocessing step. Since original data is in a range between 0 and 255, the images are normalized
to have zero mean and one standard deviation.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| ReLU                  |                                               |
| Max pooling           | 2x2 stride, valid padding, 5x5x16             | 
| Fully connected		| Input size: 400, output size: 120             |
| ReLU                  |                                               |
| Fully connected       | Input size: 120, output size: 84              |
| Softmax				| Input size: 84, output size: 43 |
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with cross entropy as a loss function. The used hyperparameters are the following:
-learning rate: 0.002
-number of epochs: 30
-batch size: 256

Dropout is used for two fully connected layers:
-keeping probability: 0.4

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.955
* test set accuracy of 0.944

I. As a first step LeNet architecture was used for non-processed dataset. As it was expected the network did not perform well.
II. After normalizing input dataset the network had over 98% accuracy on a training dataset, but accuracy on a validation set was below 90%.
Based on this observation we could say that the network was overfitting.
III. After adding dropout in two fully connected layers it was possible to hit over 92% accuracy.
- At this step different convnet architectures were tried:
	- more convolutional layers
	- different filter sizes
	- padding: same and valid
However, these steps did not bring any significant improvement in accuracy.
IV. Validation accuracy 95.5% and test accuracy 94.4% was achieved by tuning hyperparameters(batch size, learning rate, keep_prob) for 
initial LeNet architecture with dropout. 
	

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
[image1]: ./Signs/11.jpg "Right-of-way at the next intersection"
[image2]: ./Signs/14.jpg "Stop"
[image3]: ./Signs/22.jpg "Bumpy Road"
[image4]: ./Signs/25.jpg "Road Work"
[image5]: ./Signs/31.jpg "Wild Animals Crossing"

The first and fourth images are expected to be simple to classify. Since signs are slighly rotated the second and third images might be
hard to classify. The last image is mirrored version of the signs that in the training dataset. Therefore, it could be hard to classify. 
Note that if we did augment the training data, two above problems could be solved.    

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way    		| Right-of-way   								| 
| Stop     			    | Stop										    |
| Bumpy Road		    | Bicycles crossing                             |
| Road Work	      		| Road Work					 					|
| Wild Animals Crossing | Right-of-way     							    |

The model was able to predict 3 out of 5 images(accuracy 60%). The results show that augmenting the training data might be very helpful.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99        			| Right-of-way  								| 
| 0.99     				| Stop										    |
| 0.95					| Bicycles crossing								|
| 1.00	      			| Road Work						 				|
| 0.65				    | Right-of-way      							|

For the first, second and fourth images the network made correct predictions with certainty almost 100%. 
For the third image the network made a wrong prediction with certainty more than 95%. The second softmax probability was a true image sign,
however it had only about 3% probability. 
For the last image the network gave 65% and 34% probabilities for "right-of-way at the next intersection" and "children crossing" traffic 
signs correspondingly. However, the true image is "wild animals crossing" traffic sign. 
  



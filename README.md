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

#### Basic summary of the dataset.

The pandas library has been used to calculate summary statistics of the traffic signs dataset:
* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: 32x32
* The number of unique classes/labels in the data set is: 43

### Design and Test a Model Architecture

#### 1. Preprocessing
An original data is in a range between 0 and 255. Therefore the images have been normalized to have zero mean and one standard deviation.


#### 2. Model architecture
The final model consisted of the following layers:

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
 
#### 3. Model Training
To train the model, the Adam optimizer with a loss function = cross entropy has been used. The used hyperparameters are the following:
* learning rate: 0.002
* number of epochs: 30
* batch size: 256
Dropout has been used for two fully connected layers:
* keeping probability: 0.4

#### 4. Results
My final model results were:
* training set accuracy: 0.995
* validation set accuracy: 0.955
* test set accuracy: 0.944


* As the first step LeNet architecture has been used for non-processed dataset. As it was expected the network did not perform well.
* After normalizing input dataset the network had over 98% accuracy on a training dataset, but accuracy on a validation set was below 90%. Based on this observation we could say that the network was overfitting.
* After adding dropout in two fully connected layers it was possible to hit over 92% accuracy. At this step different convnet architectures have been tried:
	* more convolutional layers
	* different filter sizes
	* padding: same and valid
* However, these steps did not bring any significant improvement in accuracy. Validation accuracy 95.5% and test accuracy 94.4% have been achieved by tuning hyperparameters(batch size, learning rate, keep_prob) for initial LeNet architecture with dropout. 
	

### Test a Model on New Images

#### 1. Results for five German traffic signs found on the web
Here are five German traffic signs that I found on the web:
* <img src="https://github.com/Babiole77/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic%20Signs/11.jpg" width="40" height="40" /> Right-of-way at the next intersection
* <img src="https://github.com/Babiole77/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic%20Signs/14.jpg" width="40" height="40" /> Stop Sign
* <img src="https://github.com/Babiole77/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic%20Signs/22.jpg" width="40" height="40" /> Bumpy Road
* <img src="https://github.com/Babiole77/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic%20Signs/25.jpg" width="40" height="40" /> Road Work
* <img src="https://github.com/Babiole77/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic%20Signs/31.jpg" width="40" height="40" /> Wild Animals Crossing

The first and fourth images are expected to be simple to classify. Since signs are slighly rotated the second and third images might be hard to classify. The last image is mirrored version of the signs that in the training dataset. Therefore, it could be hard to classify. Note that if we did augment the training data, two above problems could be solved.    

#### 2. Prediction Results
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way    		| Right-of-way   								| 
| Stop     			    | Stop										    |
| Bumpy Road		    | Bicycles crossing                             |
| Road Work	      		| Road Work					 					|
| Wild Animals Crossing | Right-of-way     							    |

The model was able to predict 3 out of 5 images(accuracy 60%). The results show that augmenting the training data might be very helpful.

#### 3. Certainty of softmax probabilities
The code for making predictions in my final model is located in the 16th cell of the Ipython notebook.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99        			| Right-of-way  								| 
| 0.99     				| Stop										    |
| 0.95					| Bicycles crossing								|
| 1.00	      			| Road Work						 				|
| 0.65				    | Right-of-way      							|

For the first, second and fourth images the network made correct predictions with certainty almost 100%. For the third image the network made a wrong prediction with certainty more than 95%. The second softmax probability was a true image sign, however it had only about 3% probability. For the last image the network gave 65% and 34% probabilities for "right-of-way at the next intersection" and "children crossing" traffic signs correspondingly. However, the true image is "wild animals crossing" traffic sign. 
  



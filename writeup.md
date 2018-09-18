## CarND Vehicle Detection

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/vehicles.png
[image2]: ./output_images/non-vihecles.png
[image3]: ./output_images/HOG_example.jpg
[image4]: ./output_images/sliding_windows.jpg
[image5]: ./output_images/sliding_window.jpg
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.
You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 79 through 145 of the file called `feature_extraction.py`). I am not only extracted HOG features but also Spatial Binning and Color Histograms to got sufficient features to train the algorithms.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is a sample of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and extracted the HOG features,
trained a linear SVM using these features. The parameters that got the best 
validation accuracy be settled on.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

When it comes to selecting a machine learning algorithm for vehicle tracking problem, We had two concerns namely: accuracy and speed. So We picked Linear SVM classifier because it provides a reasonable trade-off between speed and accuracy.

I trained a linear SVM using combined features of HOG, Spatial Binning and Color Histograms extracted from vehicles and non-vehicles images. 

I use `sklearn.preprocessing.StandardScaler()` to normalize my feature vectors for training the classifier and used Grid Search hyper-parameter optimization method `klearn.model_selection.GridSearchCV` .
these are the optimized hyper-parameter values：`C = 0.08, penalty = 'l2', loss = 'hinge'`.

I split the dataset in 80% training data and 20% test data and got the 98.64% accuracy on the test data.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for sliding window search is contained in lines 212 through 265 of the file called `feature_extraction.py`).
I tried several window sizes, overlaps, and search areas. But, finally, we picked the following window size, overlap, and search areas which perform better on images and videos.
```
xy_window = (96, 96)
xy_overlap = (0.75, 0.75)
y_start_stop = [400, 600]
x_start_stop = [None, None]

```
Following image shows the locations of the search windows used by our detection system.

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is an example image:

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/processed_project_video.mp4)

To speed things up, I create a function `find_cars` which is contained in lines 317 through 404 of the file called `feature_extraction.py` to extract HOG features just once for the entire region of interest in each full image / video frame and subsample that array for each sliding window. 

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project, I used HOG, special binnings and color histograms for feature extraction. In addition to that Linear SVM classifier was used for object detection.

Though vehicle detection works well, the key drawback of this system is its performance.

Deep learning techniques using convolution neural network can help predict different type of objects ie cars, pedestrians etc more efficiently combined with transfer learning where trained weights can be utilized ie Yolo to improve the model and predictions in different conditions.

in the future, I'm planning to evaluate those new deep learning based system for vehicle tracking.



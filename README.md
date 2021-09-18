# Dog Breed Classifier
`The capstone project for Udacity’s Data Scientist Nanodegree Program`

### Table of Contents
1. [Project Overview](#overview)
2. [Problem Statement](#statement)
3. [Performance Metric](#metric)
4. [Data Exploration and Visualization](#eda)
5. [Data Preprocessing](#preprocess)
6. [Human Detector](#human)
7. [Dog Detector](#dog)
8. [Dog Breed Classifier](#breed)
9. [List of Dependencies](#dependency)
10. [Instructions to use the repository](#instructions)
11. [File Descriptions](#desc)
12. [Results](#results)
13. [Conclusion](#conc)
14. [Tips to improve the performance](#improve)
15. [Licensing, Authors, and Acknowledgements](#licensing)


## Project Overview<a name="overview"></a>
In this project, I have implemented an `end-to-end deep learning pipeline` that can be used **within a web or mobile app** to process real-world, user-supplied images. The pipeline will accept any user-supplied image as input and will predict whether a dog or human is present in the image. If a dog is detected in the image, it will provide an estimate of the dog’s breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 


## Problem Statement<a name="statement"></a>
In this project, I am provided with RGB images of humans and dogs and asked to design and implement an algorithm that can detect humans (human detector) or dogs (dog detector) in the images. After detecting a human or dog, the algorithm further needs to predict the breed of the dog (if the dog is detected) and the most resembling dog breed (if a human is detected). If neither is detected in the image, the algorithm should ask the user to input another image containing either dog or human.


## Performance Metric<a name="metric"></a>
To evaluate the performance of my algorithm, I used classification accuracy as the performance metric. All three deep learning models human detector, dog detector, and dog breed classifier were evaluated using the accuracy that these models have obtained in classifying the images.

Accuracy is a reasonable choice of performance metric for these models. This is because the human detector model is provided with 100 images of humans and 100 images of dogs (balanced data) to evaluate its accuracy. Similarly, the dog detector is provided with 100 images of each human and dog to evaluate its accuracy. 


## Data Exploration and Visualization<a name="eda"></a>
The dog breed dataset contains 8351 dog images with 133 dog breed categories. The dataset is not perfectly balanced. The mean number of images in each class is around 50. But there are few classes in the dataset that have less than 30 images while there are some classes that have more than 70 images. This small imbalance in data could pose a problem in training the dog breed classifier model. But this could be taken care of by over-sampling the minority classes or under-sampling the majority classes and data augmentation methods.


## Data Preprocessing<a name="preprocess"></a>
All CNN models in Keras require a 4D array/tensor as input with shape (batch_size, image_height, image_width, num_channels). The shape of each image needs to be the same for training the CNN model in batches. Therefore the input data for the dog detector model and dog breed classifier model needs to be reshaped so that all the images have the same shape.

Getting the 4D tensor ready for any pre-trained CNN model in Keras, requires some additional processing. First, the RGB image is converted to BGR by reordering the channels. All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as [103.939, 116.779, 123.68] and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image.


## Human Detector<a name="human"></a>
I used the pre-trained Haar cascade face detector model from the OpenCV library to determine if a human is present in the image or not.


## Dog Detector<a name="dog"></a>
To detect the dogs in the images, I have used a pre-trained ResNet-50 model. This model has been trained on ImageNet, a very large and popular dataset used for image classification and other vision tasks.


## Dog Breed Classifier<a name="breed"></a>
I build a simple CNN model from scratch and this model is neither too deep nor too shallow. It has five blocks of Conv2D layer followed by MaxPooling2D layer. I added a dropout layer after every two blocks of Conv2D and MaxPooing2D layers to avoid overfitting. This model didn't perform well and achieved only 5% accuracy on the test dataset.

I used six different models with pre-trained weights to classify dog breeds. The models include VGG16, VGG19, InceptionV3, ResNet50, EfficientNetB4 and Xception. Of all the models trained, the EfficientNetB4 model performed the best on the validation dataset. It achieved an accuracy of 91% on the validation data. Trained model weights are stored in `EfficientNetB4_trained_weights` folder. The accuracy of other models was below 83% on the validation data. 


## List of Dependencies<a name="dependency"></a>
The `requirements folder` list all the libraries/dependencies required to run this project.


## Instructions to use the repository<a name="instructions"></a>
1. Clone this github repository.
`git clone https://github.com/Ankit-Kumar-Saini/Dog_Breed_Classifier`

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and prepare image label pairs for training the model.

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Unzip the folder and prepare images for the face detector model.


## File Descriptions<a name="desc"></a>
1. The `haarcascades folder` contains the pre-trained weights in the `xml file format` to use with the OpenCv face detector class that has been used in this project. 

2. The `test_images folder` contains the sample images that are used to test the predictions of the final algorithm in this project.

3. The `results folder` contains the results of the algorithm tested on the test images. These are used for the purpose of quick demonstration in the results section below.

4. The `extract_bottleneck_features.py file` contains the code to use pre-trained imagenet models as **feature extractors** for transfer learning.

5. The `dog_app.ipynb file` is the main file for this project. It is a jupyter notebook containing code of face detector, dog detector and dog breed classifier models. The final algorithm that uses all these three models to make predictions is also implemented in this notebook.


## Results<a name="results"></a>
The step by step explanation of the project can be found at the post available [here](https://ankitsaini1729.medium.com/dog-breed-classifier-using-cnns-72c33ce891c6).

`Some visualizations of the predictions made by the algorithm on test images`

![alt text](https://github.com/Ankit-Kumar-Saini/Dog_Breed_Classifier/blob/main/results/dog.PNG) 

![alt text](https://github.com/Ankit-Kumar-Saini/Dog_Breed_Classifier/blob/main/results/human_1.PNG) 

![alt text](https://github.com/Ankit-Kumar-Saini/Dog_Breed_Classifier/blob/main/results/bridge.PNG) 

![alt text](https://github.com/Ankit-Kumar-Saini/Dog_Breed_Classifier/blob/main/results/human_2.PNG) 

![alt text](https://github.com/Ankit-Kumar-Saini/Dog_Breed_Classifier/blob/main/results/taj_mahal.PNG) 


## Conclusion<a name="conc"></a>
This project serves as a good starting point to enter into the domain of deep learning. Data exploration and visualizations are extremely important before training any Machine Learning model as it helps in choosing a suitable performance metric for evaluating the model. CNN models in Keras need image data in the form of a 4D tensor. All images need to be reshaped into the same shape for training the CNN models in batch. 

Building CNN models from scratch is extremely simple in Keras. But training CNN models from scratch is computationally expensive and time-consuming. There are many pre-trained models available in Keras (trained on ImageNet dataset) that can be used for transfer learning.

The most interesting thing to note is the power of transfer learning to achieve good results with small computation. It works well when the task is similar to the task on which the pre-trained model weights are optimized.


## Tips to improve the performance<a name="improve"></a>
1. Get more images per class
2. Make the dataset balanced
3. Use image augmentation methods such as CutOut, MixUp, and CutMix
4. Use VAEs/GANs to generate artificial data
5. Use activation maps to interpret the model predictions
6. Use deep learning-based approaches to detect human faces (MTCNN)


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to Udacity for the data and python 3 notebook.





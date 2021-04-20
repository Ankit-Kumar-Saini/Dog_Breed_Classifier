# Dog Breed Classifier
`The capstone project for Udacity’s Data Scientist Nanodegree Program`

### Table of Contents
1. [Project Overview](#overview)
2. [Dependencies](#dependency)
3. [Instructions to use the repository](#instructions)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)


## Project Overview<a name="overview"></a>
In this project, I have implemented an `end-to-end deep learning pipeline` that can be used **within a web or mobile app** to process real-world, user-supplied images. The pipeline will accept any user-supplied image as input and will predict whether a dog or human is present in the image. If a dog is detected in the image, it will provide an estimate of the dog’s breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 


## List of Dependencies<a name="dependency"></a>
The `requirements folder` list all the libraries/dependencies required to run this project.


## Instructions to use the repository<a name="instructions"></a>
1. Clone this github repository.
`git clone https://github.com/Ankit-Kumar-Saini/Dog_Breed_Classifier`

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and prepare image label pairs for training the model.

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Unzip the folder and prepare images for the face detector model.


## File Descriptions <a name="files"></a>
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


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to Udacity for the data and python 3 notebook.





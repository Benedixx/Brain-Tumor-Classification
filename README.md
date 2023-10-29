# Brain Tumor Multi-Classification using TensorFlow and Keras

## Project Overview

This project aims to classify different types of brain tumors using deep learning techniques implemented with TensorFlow Keras. The goal is to create a robust multi-classification model that can accurately classify brain images into various tumor categories, such as glioma, meningioma, pituitary, and no tumor.

## Dataset

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data). It consists of brain MRI images with labeled tumor types. Ensure that you have downloaded and preprocessed the dataset as required before running the code or you can use the dataset that i provided on this repository.

Dataset contains :<br>
number of glioma training data: 702 <br>
number of meningioma training data: 704<br>
number of training data without tumors: 280<br>
number of pituitary training data: 576<br>

number of glioma validation data: 199<br>
number of meningioma validation data: 209<br>
number of without tumors validation data: 158<br>
number of pituitary validation data: 268<br>


## Model Architecture

The deep learning model is built using the TensorFlow Keras libraries. The architecture typically involves:

- Preprocessing the MRI images (resizing, normalization, augmentation, etc.).
- Creating a Convolutional Neural Network (CNN) with several convolutional and pooling layers and adding dense layers with 4 output.
- Compiling the model with an appropriate loss function and optimizer.
- Training the model on the dataset.

You can find the detailed architecture and code in the Jupyter Notebook or Python script provided in this repository.

## Training Result
### training and validation acccuracy graph
![image](https://github.com/Benedixx/Brain-Tumor-Classification/assets/97221880/4575d336-9112-495a-9da5-e39269c991e0)

### training and validation acccuracy on last epoch
```bash
Epoch 25/25
20/20 [==============================] - 1s 60ms/step - loss: 0.6170 - accuracy: 0.7677 - val_loss: 0.8855 - val_accuracy: 0.7083 - lr: 3.1250e-05
```
## Improvement

### Add predict input data

The model can be used but there is no input predict after the model training, you can contribute to add the input predict.

### Need to improve the accuracy
The model accuracy can be improved, but since i have another deadline :3 and it takes a long time i have to temporarily stop. UwU

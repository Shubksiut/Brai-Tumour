# Brain Tumor Detection with CNN

## Project Overview
This project uses a Convolutional Neural Network (CNN) to detect brain tumors from medical images. The dataset includes images of normal and tumor brain cells. The model is trained to classify whether an input image contains a tumor or not.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Steps Involved](#steps-involved)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Data Visualization](#2-data-visualization)
  - [3. Model Building](#3-model-building)
  - [4. Model Training](#4-model-training)
  - [5. Model Evaluation](#5-model-evaluation)
  - [6. Prediction](#6-prediction)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

## Introduction
The goal of this project is to create an effective model that can accurately classify brain MRI images into tumor and non-tumor categories using deep learning techniques.

## Dataset
The dataset comprises images of normal and tumor brain cells. The images are divided into two directories: 
- **Normal Cells**
- **Tumor Cells**

## Project Structure
```
Brain-Tumor-Detection/
│
├── data/
│   ├── no/             # Normal brain cell images
│   └── yes/            # Tumor brain cell images
│
├── models/             # Saved models
│
├── notebooks/          # Jupyter notebooks for exploration and experiments
│
├── src/                # Source code
│
└── README.md           # Project overview and instructions
```

## Steps Involved

### 1. Data Preparation
- **Loading Data**: Load the images from the directories.
- **Labeling Data**: Assign labels to images (0 for normal, 1 for tumor).
- **Preprocessing**: Resize images and convert them to a consistent format.

### 2. Data Visualization
- **Example Images**: Display example images from each category.
- **Label Distribution**: Visualize the distribution of normal and tumor images.

### 3. Model Building
- **CNN Architecture**: Define a CNN with layers for convolution, pooling, and fully connected layers.
- **Compilation**: Compile the model with an appropriate optimizer and loss function.

### 4. Model Training
- **Training**: Train the model on the training dataset.
- **Validation**: Validate the model on a subset of the training data.

### 5. Model Evaluation
- **Accuracy and Loss**: Plot training and validation accuracy and loss.
- **Test Set Evaluation**: Evaluate the model on the test dataset to determine its generalization capability.

### 6. Prediction
- **New Image Prediction**: Predict the category of a new image using the trained model.

## Results
- **Accuracy**: The model achieved an accuracy of 88.24% on the test dataset.
- **Visualization**: Training and validation accuracy and loss were visualized to monitor performance.

## Conclusion
The CNN model successfully classifies brain MRI images into tumor and non-tumor categories with high accuracy. This model can assist in the early detection of brain tumors, potentially leading to better treatment outcomes.

## Acknowledgements
This project is inspired by the need for automated medical image classification to support healthcare professionals. Special thanks to the contributors of the open-source dataset used in this project.

---

Feel free to explore the code, experiment with the dataset, and contribute to improving the model! For any questions or suggestions, please open an issue or contact the project maintainers.

---

**Maintainer**: Shubhamkumar Singh

---

Thank you for visiting! 🌟

### README for Brain Tumor Detection Project

---

## Brain Tumor Detection Using Convolutional Neural Network

This project aims to classify brain MRI images into two categories: normal and tumor. The model is built using a Convolutional Neural Network (CNN) in TensorFlow and Keras.

### Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Image Paths](#image-paths)
- [Loading and Preprocessing Data](#loading-and-preprocessing-data)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

### Introduction

Brain tumors are life-threatening and require accurate and early detection for effective treatment. This project leverages deep learning techniques to classify brain MRI images as either normal or tumor-affected, providing a tool to assist medical professionals in diagnosis.

### Dataset

The dataset used for this project consists of MRI images categorized into two classes:
- **Normal**: Images without a tumor.
- **Tumor**: Images with a tumor.

### Libraries Used

This project utilizes the following libraries:

- **os**: For handling file and directory operations.
- **numpy**: For numerical operations.
- **pandas**: For data manipulation and analysis.
- **matplotlib**: For plotting images and graphs.
- **Pillow**: For image processing.
- **scikit-learn**: For splitting the dataset.
- **seaborn**: For visualizing data distributions.
- **tensorflow**: For building and training the neural network.
- **opencv-python**: For image handling and preprocessing.

### Image Paths

The dataset is organized into two directories:

- **normal_path**: Directory containing images of normal brain cells.

- **tumor_path**: Directory containing images of tumor brain cells.

### Loading and Preprocessing Data

1. **Load Filenames**:
    - Load the filenames of the images from the respective directories.

2. **Create Labels**:
    - Assign labels to the images: 0 for normal and 1 for tumor.

3. **Load and Preprocess Images**:
    - Resize images to 128x128 pixels and convert to RGB.

### Model Architecture

The CNN model is built with the following layers:

- **InputLayer**: Input shape of (128, 128, 3)
- **Conv2D**: 64 filters, kernel size of (3, 3), activation function 'relu'
- **MaxPooling2D**: Pool size of (2, 2)
- **Flatten**
- **Dense**: 128 units, activation function 'relu'
- **Dense**: 1 unit, activation function 'sigmoid' (for binary classification)

### Training

The model is trained with the following configurations:

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 50
- **Validation Split**: 10%

### Evaluation

To evaluate the model on the test set:

1. **Split the Dataset**:
    - Split the dataset into training and testing sets.

2. **Normalize the Pixel Values**:
    - Normalize pixel values to be between 0 and 1.

3. **Evaluate the Model**:
    - Evaluate the model on the test set.

### Prediction

To make a prediction on a new image:

1. **Load and Preprocess the Image**:
    - Resize and normalize the image.

2. **Make Prediction**:
    - Use the trained model to predict the label.

### Results

- **Training Accuracy**: Achieved high accuracy on the training set.
- **Validation Accuracy**: Consistent performance on the validation set.
- **Test Accuracy**: Demonstrated robust accuracy on the test set.

### Contributing

Contributions are welcome! Please fork the repository and submit pull requests.

### License

This project is licensed under the MIT License.

---

Feel free to customize this README file further to fit your specific project details and structure.

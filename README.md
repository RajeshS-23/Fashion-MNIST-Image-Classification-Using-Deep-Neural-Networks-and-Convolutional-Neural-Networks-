# Fashion-MNIST Image Classification using Deep Learning

This project demonstrates image classification on the Fashion-MNIST dataset using **Deep Neural Networks (DNN)** and **Convolutional Neural Networks (CNN)** implemented with **TensorFlow and Keras**.  
The goal is to accurately classify grayscale images of clothing items into one of ten predefined categories.



## Dataset Overview

**Fashion-MNIST** is a dataset of 70,000 grayscale images (28×28 pixels) of fashion products, divided into:
- **60,000 training images**
- **10,000 test images**

### Classes:
| Label | Class Name |
|------|------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |



## Models Implemented

### 1️⃣ Fully Connected Neural Network (DNN)
- Flatten layer
- Dense layer with ReLU activation
- Output layer with Softmax
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy

### 2️⃣ Convolutional Neural Network (CNN)
- Multiple Conv2D layers with ReLU
- MaxPooling layers
- Fully connected Dense layers
- Output layer for classification
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy



## Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**



## Training & Evaluation

- Data normalization (pixel values scaled to 0–1)
- Training with validation split
- Accuracy and loss visualization
- Model evaluation on test dataset
- Single image prediction with confidence score



## Visualizations Included

- Sample training and test images
- Accuracy vs Validation Accuracy graph
- Loss vs Validation Loss graph
- Prediction result visualization with:
  - Actual label
  - Predicted label
  - Model confidence



## Sample Prediction Output

The model predicts the class of a test image and displays:
- The input image
- Actual class label
- Predicted class label
- Prediction confidence



## Future Improvements

- Add Dropout layers to reduce overfitting
- Experiment with Batch Normalization
- Use data augmentation
- Try deeper CNN architectures
- Compare performance with transfer learning models



##Conclusion

This project showcases the effectiveness of CNNs over traditional dense networks for image classification tasks and provides hands-on experience with building, training, and evaluating deep learning models using TensorFlow.



**Rajesh S**  
Aspiring Machine Learning Engineer  

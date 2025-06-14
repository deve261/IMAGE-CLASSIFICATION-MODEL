# IMAGE-CLASSIFICATION-MODEL

COMPANY: CODTECH IT SOLUTIONS


NAME: YELLANKI DEVENDRA


INTERN ID: CT04DG224


DOMAIN: MACHINE LEARNING


DURATION: 4 WEEKS


MENTOR: NEELA SANTOSH


PROJECT DESCRIPTION: Image Classification App Using Convolutional Neural Network (CNN) and Tkinter GUI
This project is a fully functional, interactive desktop application designed to classify images into one of ten categories using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. It combines deep learning with a user-friendly graphical interface developed in Python using the Tkinter library, making it accessible for both technical and non-technical users. The classifier is built using TensorFlow and Keras, two of the most widely used frameworks for deep learning.

The application loads a pre-trained model (cnn_model.h5) or trains one if not available, then allows users to upload any image from their local device. It predicts the class of the uploaded image and displays both the image and the predicted category on the interface.

1. Dataset and Model Architecture
The underlying model has been trained on the CIFAR-10 dataset, a standard computer vision dataset that contains 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The categories include:

Airplane

Automobile

Bird

Cat

Deer

Dog

Frog

Horse

Ship

Truck

The CNN architecture follows a relatively simple yet effective design pattern, suitable for image classification tasks:

Input Layer: Accepts 32x32x3 RGB images.

Convolutional Layers: Three Conv2D layers with ReLU activation extract spatial features from the image.

Pooling Layers: MaxPooling layers reduce the spatial dimensions, improving computational efficiency.

Flatten and Dense Layers: The final layers convert the 2D feature maps into a 1D vector, which is fed into two Dense layers. The output layer uses a softmax activation to assign probabilities across the 10 categories.

The model is compiled using the Adam optimizer and trained using sparse categorical crossentropy loss for multi-class classification. It is trained for 10 epochs with a validation split of 10%.

2. Model Training and Saving
If the model file cnn_model.h5 does not exist in the working directory, the script automatically initiates training using the CIFAR-10 dataset. Once training is completed, the model is saved to disk for reuse. This reduces the startup time for future uses and ensures the app remains responsive.

Model evaluation on the test set is also printed to the console to provide insight into the classifier's performance.

3. Real-Time Prediction Interface
Once the model is loaded, the user interface—developed with Tkinter—is launched. The GUI includes:

A "Select Image" button that opens a file dialog for uploading an image.

An image preview window that resizes and displays the selected image.

A prediction label that shows the predicted class (e.g., "Prediction: dog").

When a user selects an image, the app processes it as follows:

It resizes the image to 32x32 pixels (the input size required by the CNN).

Converts the image into a NumPy array and normalizes the pixel values.

Feeds the processed image into the loaded model.

Displays the top predicted class based on the model’s output.

This setup enables real-time prediction and feedback, making the application highly interactive and user-friendly.

4. Applications and Use Cases
While the current model is trained on CIFAR-10, this framework can be extended to other datasets for real-world applications such as:

Classifying animals, vehicles, or everyday objects.

Building educational tools for learning AI and image recognition.

Rapid prototyping of vision-based software for mobile or desktop platforms.

With enhancements, this project could be scaled to support more sophisticated models like ResNet or MobileNet and support real-time webcam input or batch image classification.

5. Key Technologies Used
Python: Programming language.

TensorFlow / Keras: For deep learning model creation and training.

NumPy: For efficient image manipulation.

PIL (Pillow): To load and resize images.

Tkinter: For building the graphical user interface.

#OUTPUT :

![Image](https://github.com/user-attachments/assets/fd2f19f9-1f76-4ce0-a22b-516b1d6a99ac)
![Image](https://github.com/user-attachments/assets/684ccd3e-638e-423b-a4fd-0be5719d70ce)




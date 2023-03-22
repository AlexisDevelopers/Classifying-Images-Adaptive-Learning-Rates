This code is an implementation of a convolutional neural network (CNN) model for classifying images from the MNIST dataset. The objective is to train a model capable of recognizing handwritten digits with high precision.

In the first block of code, the libraries needed to build the model and display the results are imported. TensorFlow, a deep learning library, is imported along with other common libraries like NumPy, Matplotlib, and Scikit-learn.

Next, the MNIST dataset is loaded, containing images of handwritten digits, along with the corresponding labels. The images are scaled to have values between 0 and 1, and the labels are transformed into a hot vector.

The data set is divided into two parts: one for training and one for evaluation. The images are transformed into tensors and the model is configured.

The first model, called "model_uno", uses a constant learning rate (0.5) and a root mean square loss function. The model is trained for 25 epochs and the training history is saved for later viewing.

The second model, "model_dos", uses a learning rate that decreases as the number of epochs increases. This is accomplished using a learning rate scheduler that is called after each epoch. The scheduler uses a feature that reduces the learning rate by 1% after each epoch. Again, the model is trained for 25 epochs and the training history is saved for later analysis.

The third model, "model_tres", uses a cyclic learning rate. This approach adjusts the learning rate between a minimum value and a maximum value during training instead of keeping it constant or gradually decreasing it. The TensorFlow Addons "CyclicalLearningRate" optimizer is used to tune the cyclic learning rate. The model is trained for 25 epochs and the training history is also saved for later viewing.

Finally, the evolution of the learning rate in the third model is shown.
# Deep Learning

## Description

Assignments for Deep Learning at Korea University (2021 Fall)

## Assignments

### 1. Linear Regression

Performed linear regression to estimate the linear model with noise, which is equivalent to minimizing the mean square error (MSE) between the estimated values and the actual value.

### 2. Various Classifiers

Implemented three classifiers - k-Nearest Neighbor (kNN), Support Vector Machine (SVM), and Softmax - and measured the test accuracy of each classifier after training them on a randomly-generated dataset. A vectorization trick on matrices was a key point to enhance the performance.

### 3. Neural Network

Implemented a 2-layer neural network composed of linear, sigmoid, and softmax operations and trained it with the cross entropy loss to mimic an XOR operation. It was challenging to derive the derivates of output w.r.t input (based on the chain rule). 

### 4. Basic Layers

Implemented convolutional and maxpool layers by writing codes for forward and backward (backpropagation) operations and utilized some matplotlib functions to plot useful data. Since the inputs were given in form of a mini-batch, it requires a bit of effort to handle dimensions.

### 5. Convolutional Neural Networks

Fully implemented a convolutional neural network by writing codes for forward and backward operations of each sub-layer (e.g., convolutional, fully-connected, softmax, etc.) and the network.

### 6. Frameworks

With PyTorch, implemented a ResNet architecture from scratch and trained it on Google Colab to utilize GPU. When the network was trained for 5 epochs on the CIFAR-10 dataset, its total accuracy was over 76%.

# BackPropagation
Deep Learning Algorithm 

Backpropagation is a method used in artificial neural networks to calculate a gradient that is needed in the calculation of the weights to be used in the network

The goal of the backpropagation training algorithm is to modify the weights of a neural network in order to minimize the error of the network outputs

1. Implement the Back-Propagation learning algorithm on a multi-layer neural networks, which can be able to classify a stream of input data to one of a set of predefined classes.

Use the iris data in both your training and testing processes. (Each class has 50 samples: train NN with the first 30 non-repeated samples, and test it with the remaining 20 samples)

2. After training
Test the classifier with the remaining 20 samples of each selected classes and find confusion matrix and compute overall accuracy.

---------------------------------------------------------------------------------------------------------------------

1. User Input:
• Enter number of hidden layers
• Enter number of neurons in each hidden layer
• Enter learning rate (eta)
• Enter number of epochs (m)
• Add bias or not (Checkbox)
• Choose to use Sigmoid or Hyperbolic Tangent sigmoid as the activation     function

2. Initialization:
• Number of features = 4.
• Number of classes = 3.
• Weights + Bias = small random numbers

3. Classification:
• Sample (single sample to be classified).

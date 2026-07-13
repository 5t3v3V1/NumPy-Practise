import numpy as np

M = np.array([[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])
y_true = np.array([[0],
                  [1],
                  [1],
                  [0]])
learning_rate = 0.01
epsilon = 1e-15

class Layer:
    def __init__(self, input, output):
        self.weights = np.random.randn(input, output)
        self.bias = np.zeros(output)

    def forward(self, matrix):
        self.inputs = matrix
        
        return matrix @ self.weights + self.bias
    
    def backward(self, gradient):
        self.dweights = self.inputs.T @ gradient / len(self.inputs)
        self.dbias = np.mean(gradient, axis = 0)

        return gradient @ self.weights.T
    
    def update(self, rate):
        self.weights -= rate * self.dweights
        self.bias -= rate * self.dbias
    
class ReLU:
    def forward(self, x):
        self.inputs = x
        return np.maximum(0, x)
    
    def backward(self, gradient):
        return gradient * (self.inputs > 0)
    
class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, gradient):
        return gradient * (1 - self.output ** 2)
    
class Sigmoid:
    def forward(self, x):
        self.output = 1/(1 + np.exp(-x))
        return self.output

relu = ReLU()
tanh = Tanh()
sigmoid = Sigmoid()
hidden = Layer(2, 4)
output = Layer(4, 1)

for i in range(1000):
    z_hidden = hidden.forward(M)

    a_hidden = tanh.forward(z_hidden)

    z_pred = output.forward(a_hidden)
    y_pred = sigmoid.forward(z_pred)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    dloss = y_pred - y_true

    doutput = output.backward(dloss)
    dtanh = tanh.backward(doutput)
    hidden.backward(dtanh)
    
    output.update(learning_rate)
    hidden.update(learning_rate)

    if i % 100 == 0:
        predictions = (y_pred >= 0.5).astype(float)
        print(np.unique(predictions, return_counts = True))
        accuracy = np.mean(predictions == y_true)
        print(f"Step {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, a1: {np.mean(a_hidden[:,0]):.4f}, a2: {np.mean(a_hidden[:,1]):.4f}, a3: {np.mean(a_hidden[:,2]):.4f}, Prediction: {np.array2string(y_pred, precision = 2, suffix = '', separator = ',')}")

predictions = (y_pred >= 0.5).astype(float)
accuracy = np.mean(predictions == y_true)
print(accuracy)


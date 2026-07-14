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


class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers

    def forward(self, matrix):
        inputs = matrix
        for layer in self.layers:
            inputs = layer.forward(inputs)
        
        self.y_pred = inputs
        return self.y_pred

    def backward(self, true, criterion):
        loss, dloss = criterion.forward(self.y_pred, true)
        gradient = dloss
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return loss
    
    def update(self, learning_rate):
        for layer in reversed(self.layers):
            layer.update(learning_rate)

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
    
    def update(self, learning_rate):
        pass
    
class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, gradient):
        return gradient * (1 - self.output ** 2)
    
    def update(self, learning_rate):
        pass
    
class Sigmoid:
    def forward(self, x):
        self.output = 1/(1 + np.exp(-x))
        return self.output
    
    def backward(self, gradient):
        return gradient

    def update(self, learning_rate):
        pass

class BinaryCrossEntropy:
    def forward(self, y_pred, y_true):
        loss = np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        dloss = y_pred - y_true
        return loss, dloss

network = NeuralNetwork([
    Layer(2, 8),
    ReLU(),
    Layer(8, 8),
    Tanh(),
    Layer(8, 1),
    Sigmoid()
])
bce = BinaryCrossEntropy()

while True:
    try:
        print("1. Train")
        print("2. Exit")

        choice = input("Choice: ")

        if choice == "1":
            loops = int(input("Please input how many loops: "))
        
            for i in range(loops):
                y_pred = network.forward(M)

                loss = network.backward(y_true, bce)

                network.update(learning_rate)

                if i % 100 == 0:
                    predictions = (y_pred >= 0.5).astype(float)
                    print(np.unique(predictions, return_counts = True))
                    accuracy = np.mean(predictions == y_true)
                    print(f"Step {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Prediction: {np.array2string(y_pred, precision = 2, suffix = '', separator = ',')}")

            predictions = (y_pred >= 0.5).astype(float)
            accuracy = np.mean(predictions == y_true)
            print(accuracy)
        
        elif choice == "2":
            break

        else:
            print("Invalid Choice")
        
    except ValueError:
        print("Please input a number")



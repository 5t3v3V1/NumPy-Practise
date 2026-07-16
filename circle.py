import numpy as np

np.random.seed(42)

n = 1000

# Random angles
theta = np.random.rand(n) * 2 * np.pi

# Inner circle
r1 = 1 + 0.1 * np.random.randn(n)
x1 = r1 * np.cos(theta)
y1 = r1 * np.sin(theta)

# Outer circle
r2 = 3 + 0.1 * np.random.randn(n)
x2 = r2 * np.cos(theta)
y2 = r2 * np.sin(theta)

M = np.vstack([
    np.column_stack([x1, y1]),
    np.column_stack([x2, y2])
])

y_true = np.vstack([
    np.zeros((n, 1)),
    np.ones((n, 1))
])
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
        self.trainable = True
        self.weights = np.random.randn(input, output)
        self.bias = np.zeros(output)
        self.classification = f"Layer ({input} -> {output})\nWeights: ({input}, {output})\nBias: ({output})"

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
    def __init__(self):
        self.trainable = False
        self.classification = "Activation: ReLU"

    def forward(self, x):
        self.inputs = x
        return np.maximum(0, x)
    
    def backward(self, gradient):
        return gradient * (self.inputs > 0)
    
    def update(self, learning_rate):
        pass
    
class Tanh:
    def __init__(self):
        self.trainable = False
        self.classification = "Activation: Tanh"

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, gradient):
        return gradient * (1 - self.output ** 2)
    
    def update(self, learning_rate):
        pass
    
class Sigmoid:
    def __init__(self):
        self.trainable = False
        self.classification = "Activation: Sigmoid"

    def forward(self, x):
        self.output = 1/(1 + np.exp(-x))
        return self.output
    
    def backward(self, gradient):
        return gradient * self.output * (1 - self.output)

    def update(self, learning_rate):
        pass



class BinaryCrossEntropy:
    def __init__(self):
        self.classification = "Loss Function: Binary Cross Entropy"

    def forward(self, y_pred, y_true):
        loss = np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        dloss = -(y_true/y_pred - (1 - y_true)/1 - y_pred)
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
        print("-" * 40)
        print(bce.classification)
        print("-" * 40)
        print()
        for layer in network.layers:
            print(layer.classification)
            print()
            if not layer.trainable:
                print("-" * 40)
                print()
        print("-" * 40)
        print("Parameters: ")

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
                    print(f"Step {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

            predictions = (y_pred >= 0.5).astype(float)
            accuracy = np.mean(predictions == y_true)
            print(accuracy)
        
        elif choice == "2":
            break

        else:
            print("Invalid Choice")
        
    except ValueError:
        print("Please input a number")



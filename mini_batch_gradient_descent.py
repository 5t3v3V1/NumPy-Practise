import numpy as np

x = np.random.randn(300)
y = np.random.randn(300)
z = np.random.randn(300)
M = np.column_stack((x, y, z))
d_true = 4 * x + 5 * y + 2 * z
labels = np.zeros(300, dtype = int)
labels[d_true > 2] = 2
labels[(d_true <= 2) & (d_true >= -2)] = 1
labels[d_true < -2] = 0
y_true = np.eye(3)[labels]
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
        return gradient

    def update(self, learning_rate):
        pass

class Softmax:
    def __init__(self):
        self.trainable = False
        self.classification = "Activation: Softmax"

    def forward(self, x):
        shift_back = x - np.max(x, axis = 1, keepdims = True)
        exp = np.exp(shift_back)
        self.output = exp / np.sum(exp, axis = 1, keepdims = True)
        return self.output

    def backward(self, gradient):
        return gradient

    def update(self, learning_rate):
        pass 

class BinaryCrossEntropy:
    def __init__(self):
        self.classification = "Loss Function: Binary Cross Entropy"

    def forward(self, y_pred, y_true):
        loss = np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        dloss = y_pred - y_true
        return loss, dloss
    
class CategoricalCrossEntropy:
    def __init__(self):
        self.classification = "Loss Function: Categorical Cross Entropy"

    def forward(self, y_pred, y_true):
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis = 1))
        dloss = y_pred - y_true
        return loss, dloss

network = NeuralNetwork([
    Layer(3, 8),
    ReLU(),
    Layer(8, 8),
    Tanh(),
    Layer(8, 3),
    Softmax()
])

loss_function = CategoricalCrossEntropy()

while True:
    try:
        print("-" * 40)
        print(loss_function.classification)
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
            epochs = int(input("Please input how many epochs: "))
            batch_size = int(input("Please input batch size: "))
        
            for epoch in range(epochs):
                indices = np.random.permutation(len(M))
                
                m_shuffled = M[indices]
                y_true_shuffled = y_true[indices]

                for start in range(0, len(M), batch_size):

                    end = start + batch_size

                    m_batch = m_shuffled[start:end]
                    y_batch = y_true_shuffled[start:end]

                    y_pred = network.forward(m_batch)

                    loss = network.backward(y_batch, loss_function)

                    network.update(learning_rate)

                if epoch % 100 == 0:
                    predictions = np.argmax(y_pred, axis = 1)
                    print(np.unique(predictions, return_counts = True))
                    true = np.argmax(y_batch, axis=1)
                    accuracy = np.mean(predictions == true)
                    print(f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

            predictions = np.argmax(y_pred, axis=1)
            true = np.argmax(y_batch, axis=1)
            accuracy = np.mean(predictions == true)
            print(accuracy)
        
        elif choice == "2":
            break

        else:
            print("Invalid Choice")
        
    except ValueError:
        print("Please input a number")



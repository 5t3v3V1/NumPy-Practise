import numpy as np

x = np.random.randn(100)
y = np.random.randn(100)
z_true = 3 * x + 4 * y + np.random.randn(100) * 0.5
y_true = (z_true > 0).astype(float)
print(np.mean(y_true))
learning_rate = 0.01
epsilon = 1e-15

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class Neuron():
    def __init__(self):
        self.w1 = np.random.randn() 
        self.w2 = np.random.randn() 
        self.b = 0

neuron1 = Neuron()
neuron2 = Neuron()
neuron3 = Neuron()

w1 = np.random.randn()
w2 = np.random.randn()
w3 = np.random.randn()
b = np.random.randn()

for i in range(1000):
    z1 = neuron1.w1 * x + neuron1.w2 * y + neuron1.b
    z2 = neuron2.w1 * x + neuron2.w2 * y + neuron2.b
    z3 = neuron3.w1 * x + neuron3.w2 * y + neuron3.b

    a1 = relu(z1)
    a2 = relu(z2)
    a3 = relu(z3)

    

    z_pred = w1 * a1 + w2 * a2 + w3 * a3 + b
    y_pred = 1/(1 + np.exp(-z_pred))
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    dloss = y_pred - y_true

    dw1 = np.mean(dloss * a1)
    dw2 = np.mean(dloss * a2)
    dw3 = np.mean(dloss * a3)
    db = np.mean(dloss)

    da1 = dloss * w1
    da2 = dloss * w2
    da3 = dloss * w3
    
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    w3 -= learning_rate * dw3
    b -= learning_rate * db
    
    dz1 = da1 * relu_derivative(z1)
    dz2 = da2 * relu_derivative(z2)
    dz3 = da3 * relu_derivative(z3)

    neuron1.w1 -= learning_rate * np.mean(dz1*x)
    neuron1.w2 -= learning_rate * np.mean(dz1*y)
    neuron1.b -= learning_rate * np.mean(dz1)

    neuron2.w1 -= learning_rate * np.mean(dz2*x)
    neuron2.w2 -= learning_rate * np.mean(dz2*y)
    neuron2.b -= learning_rate * np.mean(dz2)

    neuron3.w1 -= learning_rate * np.mean(dz3*x)
    neuron3.w2 -= learning_rate * np.mean(dz3*y)
    neuron3.b -= learning_rate * np.mean(dz3)

    if i % 100 == 0:
        predictions = (y_pred >= 0.5).astype(float)
        print(np.unique(predictions, return_counts = True))
        accuracy = np.mean(predictions == y_true)
        print(f"Step {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, a1: {np.mean(a1):.4f}, a2: {np.mean(a2):.4f}, a3: {np.mean(a3):.4f}, w1: {w1:.4f}, w2: {w2:.4f}, w3: {w3:.4f}, b: {b:.4f}")

predictions = (y_pred >= 0.5).astype(float)
accuracy = np.mean(predictions == y_true)
print(accuracy)


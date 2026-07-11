import numpy as np

np.random.seed(42)
x = np.random.randn(100)
z_true = 3 * x + 2 + np.random.randn(100) * 0.5
y_true = (z_true > 0).astype(float)

w1 = np.random.rand() * 0.01
w2 = np.random.rand() * 0.01
b = np.random.rand() * 0.01

learning_rate = 0.01

epsilon = 1e-15

class Neuron():
    w1 = np.random.rand() * 0.01
    w2 = np.random.rand() * 0.01
    b = np.random.rand() * 0.01

for i in range(1000):
    z_pred = w * x + b
    y_pred = 1/(1 + np.exp(-z_pred))
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    dw = np.mean((y_pred - y_true) * x)
    db = np.mean(y_pred - y_true)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss:.4f}, w: {w:.4f}, b: {b:.4f}")

predictions = (y_pred >= 0.5).astype(float)
accuracy = np.mean(predictions == y_true)
print(accuracy)
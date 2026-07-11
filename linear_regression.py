import numpy as np

np.random.seed(42)
x = np.random.randn(100)
y_true = 3 * x + 2 + np.random.randn(100) * 0.5

w = np.random.rand() * 0.01
b = np.random.rand() * 0.01

learning_rate = 0.01

for i in range(1000):
    y_pred = w * x + b

    loss = np.mean((y_pred - y_true) ** 2)

    dw = (2/len(x)) * np.sum((y_pred - y_true) * x)
    db = (2/len(x)) * np.sum(y_pred - y_true)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss:.4f}, w: {w:.4f}, b: {b:.4f}")
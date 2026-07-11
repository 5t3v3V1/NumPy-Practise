import numpy as np

x = np.random.randn(100)
y = np.random.randn(100)
M = np.column_stack((x, y))
z_true = 4 * x + 5 * y + np.random.randn(100) * 0.5
y_true = (z_true > 0).astype(float).reshape(-1, 1)
print(np.mean(y_true))
learning_rate = 0.01
epsilon = 1e-15

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

w_hidden = np.random.randn(2, 3)
b_hidden = np.zeros(3)

w_output = np.random.randn(3, 1)
b_output = 0

for i in range(1000):
    z_hidden = M @ w_hidden + b_hidden

    a_hidden = relu(z_hidden)

    z_pred = a_hidden @ w_output + b_output
    y_pred = 1/(1 + np.exp(-z_pred))
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    dloss = y_pred - y_true

    dwo = a_hidden.T @ dloss / len(M)
    dbo = np.mean(dloss)

    dzh = (dloss @ w_output.T) * relu_derivative(z_hidden)
    dwh = M.T @ dzh / len(M)
    dbh = np.mean(dzh, axis = 0)
    
    w_output -= learning_rate * dwo
    b_output -= learning_rate * dbo

    w_hidden -= learning_rate * dwh
    b_hidden -= learning_rate * dbh

    if i % 100 == 0:
        predictions = (y_pred >= 0.5).astype(float)
        print(np.unique(predictions, return_counts = True))
        accuracy = np.mean(predictions == y_true)
        print(f"Step {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, a1: {np.mean(a_hidden[0][0]):.4f}, a2: {np.mean(a_hidden[1][0]):.4f}, a3: {np.mean(a_hidden[2][0]):.4f}, w1: {w_output[0][0]:.4f}, w2: {w_output[1][0]:.4f}, w3: {w_output[2][0]:.4f}, b: {b_output:.4f}")

predictions = (y_pred >= 0.5).astype(float)
accuracy = np.mean(predictions == y_true)
print(accuracy)


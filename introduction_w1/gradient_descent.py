import numpy as np


def cost_derivative_with_respect_to_w(x, y, w, b):
    """
    Partial derivative of cost function with respect to weight.
    Args:
        x, y (ndarray (m,)): Data, m examples
        w, b (scalar):       Weight and bias, scalars
    Returns
        cost (scalar):       Cost of the function with given parameters w and b
    """
    cost = 0
    m = x.shape[0]
    for i in range(m):
        cost += (w * x[i] + b - y[i]) * x[i]
    cost /= m

    return cost

def cost_derivative_with_respect_to_b(x, y, w, b):
    """
    Partial derivative of cost function with respect to bias.
    Args:
        x, y (ndarray (m,)): Data, m examples
        w, b (scalar):       Weight and bias, scalars
    Returns
        cost (scalar):       Cost of the function with given parameters w and b
    """
    cost = 0
    m = x.shape[0]
    for i in range(m):
        cost += (w * x[i] + b - y[i])
    cost /= m

    return cost

def gradient_descent(x, y, learning_rate=0.1):
    """
    Finds optimal values for weight and bias in a linear regression cost function.
    Args:
        x, y (ndarray (m,)): Data, m examples
    Returns
        w, b (scalar):       Optimal values for weight and bias parameters
    """
    # Initializes weight and bias to zero
    w = b = 0
    # Keeps track of previous weight and bias values for convergence checking
    prev_w = prev_b = -1

    while w != prev_w or b != prev_b:
        # Computes next weight value using the partial derivative
        # of the cost function with respect to weight
        tmp_w = w - learning_rate * cost_derivative_with_respect_to_w(x, y, w, b)
        # Computes next bias value using the partial derivative
        # of the cost function with respect to bias
        tmp_b = b - learning_rate * cost_derivative_with_respect_to_b(x, y, w, b)
        # Keeps tracks of previous values
        prev_w = w
        prev_b = b
        # Simultaneously updates weight and bias values
        w = tmp_w
        b = tmp_b

    return [w, b]


def get_model(x, y):
    """
    Gets the linear regression model, given a set of reference data
    Args:
        x, y (ndarray (m,)): Data, m examples
    Returns
        model (function):    Linear regression function parametrized with optimal values
    """
    # Gets optimal values for weight and bias
    [w, b] = gradient_descent(x, y)
    # Create model for linear regression function
    def model(x):
        return w * x + b

    return model

def main():
    """
    Create a model to predict prices of houses from a set of labeled data
    """
    x_train = np.array([1.0, 2.0])
    y_train = np.array([300.0, 500.0])

    model = get_model(x_train, y_train)
    prediction = model(1.2)
    expected = 340
    error = abs(prediction - expected)

    if error  < 1e-3:
        print("Model successfully predicted price")
    else:
        print(f"Model failed to predict price, it failed by {error}.")


if __name__ == '__main__':
    main()

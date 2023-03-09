import numpy as np
import matplotlib.pyplot as plt


"""
Computes the prediction of a linear model
Args:
    x (ndarray (m,)): Data, m examples 
    w,b (scalar)    : model parameters  
Returns
    y (ndarray (m,)): target values
"""
def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

"""
Predicts according to a linear model
Args:
    x (scalar):   Input data, scalar
    w,b (scalar): Weight and bias, scalars
Returns
    ŷ (scalar):   Prediction
"""
def predict(w, b, x):
    return w * x + b


def main():
    # x_train is the input variable (size in 1000 square feet)
    # y_train is the target (price in 1000s of dollars)
    x_train = np.array([1.0, 2.0])
    y_train = np.array([300.0, 500.0])

    # w is the weight of the function and b is the bias
    w = 200
    b = 100
    tmp_f_wb = compute_model_output(x_train, w, b,)

    # x is the house surface in 1000 square feet
    x_i = 1.2
    cost_1200sqft = predict(w, b, x_i)
    print(f"${cost_1200sqft:.0f} thousand")

    # Plot our model prediction
    plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

    # Plot the data points
    plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
    plt.scatter([x_i], [cost_1200sqft])

    # Set the title
    plt.title("Housing Prices")
    # Set the y-axis label
    plt.ylabel('Price (in 1000s of dollars)')
    # Set the x-axis label
    plt.xlabel('Size (1000 sqft)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
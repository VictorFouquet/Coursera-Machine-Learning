# Course 1 - Introduction to Machine Learning

## Definitions

### Supervised learning

A supervised learning algorithm is an algorithm that learns from being given right output associated to given inputs.

It learns from data labeled with the right answers.

#### Regression

A regression algorithm is a model of supervised learning that is aimed at predicting numbers Y from inputs X.

Outputs can have an infinite number of values.

#### Classification

A classification algorithm is a model of supervised learning that is aimed at predicting categories Y (can be non numerical values) from inputs X.

Outputs can have only a limited number of values.

### Unsupervised learning

An unsupervised learning algorithm is an algorithm that learns without being given the right outputs assiociated to given inputs.

The algorithm is aimed at finding structure in the data.

#### Clustering

A clustering algorithm is a model of unsupervised learning that is aimed at grouping similar data points together.

## Regression Model

### Linear regression

#### Terminology and notations

A `training set` is the data used to train the model.

In terms of notation, $x$ refers to the `input`, or `feature` variable.

$y$ refers to the `output`, or `target` variable.

$m$ refers to the number of training examples.

$(x, y)$ refers to a single training example. 

($x_i$, $y_i$) refers to the i'th training example.

$x$ is given to a function $f$, the model, that will produce a prediction $ŷ$, an estimate of the true output $y$.

$f$ can be represented as $f_w,_b(x) = wx + b$, simplified $f(x) = wx + b$.

See example running the `model_representation.py` script.

#### Cost function

$w$ and $b$ are the parameters of the function, namely its `weight` and `bias`

The predicted $ŷ$ value will be the result of an input $x$ given to $f$, written $ŷ^{(i)} = f_{w,b}(x^{(i)})$

The training process is aimed at adjusting the $w$ and $b$ parameters so that the linear model fits the data points as closely as possible, meaning minimizing the difference between $y^{(i)}$ and $ŷ^{(i)}$ values for all $x^{(i)}$.

The cost function will compute the averge square of the errors for the $y^{(i)}$ predictions given $x^{(i)}$ inputs (by convention, the number of entries in the dataset is multiplied by $2$ for neater results).

Squared error cost function:

$J(w,b) = \frac{1}{2m}{\sum_{i=1}^{m} ( ŷ^{(i)} - y^{(i)} )^2}$

We can rewrite this exaction to include our model :

$J(w,b) = \frac{1}{2m}{\sum_{i=1}^{m} ( f_{(w,b)}(x^{(i)}) - y^{(i)} )^2}$

### Gradient Descent

Gradient descent is an algorithm used to minimize functions, having any number of parameters.

#### Notation

Gradient descent wants to minimize the cost of the model: $min_{w,b}J(w,b)$

#### Algorithm

Repeat until convergence:

$w = w - \alpha\frac{\delta}{\delta w}J(w,b)$

$b = b - \alpha\frac{\delta}{\delta b}J(w,b)$

Where

- $w,b$ are the parameters to minimize
- $=$ is used there for assigment
- $\alpha$ is the learning rate
- $\frac{\delta}{\delta w}J(w,b)$ is the partial derivative of cost function $J$ with respect to $w$
- $\frac{\delta}{\delta b}J(w,b)$ is the partial derivative of cost function $J$ with respect to $b$

NB: Both updates must be done simultaneously, so the operations results will be stored in temporary $(tmp_w,tmp_b)$ values before they get assigned to $(w,b)$ respectively.

#### Calculate the derivative terms

Partial derivative of $J$ with respect to $w$ :

$\frac{\delta}{\delta w}J(w,b) = \frac{\delta}{\delta w}\frac{1}{2m}{\sum_{i=1}^{m} ( f_{(w,b)}(x^{(i)}) - y^{(i)} )^2}$

Replacing $f$ we get :

$\frac{\delta}{\delta w}J(w,b) = \frac{\delta}{\delta w}\frac{1}{2m}{\sum_{i=1}^{m} ( wx{(i)} + b - y^{(i)} )^2}$

Then applying calculus rules we get :

$\frac{\delta}{\delta w}J(w,b) = \frac{1}{m}{\sum_{i=1}^{m} ( wx{(i)} + b - y^{(i)} )x^{(i)}}$

Partial derivative of $J$ with respect to $b$ using calculus :

$\frac{\delta}{\delta b}J(w,b) = \frac{1}{m}{\sum_{i=1}^{m} ( wx{(i)} + b - y^{(i)} )}$
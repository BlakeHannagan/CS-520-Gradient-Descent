# This file was written by Blake Hannagan and Caleb Wilkins.

import numpy as np
import matplotlib.pyplot as plt
import random

#####################################################################################
#####################################################################################
# Part 1: Regression, Three Ways
#####################################################################################
#####################################################################################

# Code given in the assignment:
d = 100
n = 1000
X = np.random.normal(0, 1, size = (n, d))
a_true = np.random.normal(0, 1, size = (d, 1))
y = X.dot(a_true) + np.random.normal(0, 0.5, size = (n, 1))

#####################################################################################
# 1a: Closed Form Solution
#####################################################################################

# Get X Transpose and (X^TX)^-1 for the least squares regression.
X_t = X.T

# Multiply X_t and X
X_t_X = X_t @ X

# Get the inverse
X_t_X_inv = np.linalg.inv(X_t_X)

a = X_t_X_inv @ X_t @ y

# Calculate the total squared error of y.
errors = X.dot(a) - y
tse_closed_form = 0
for error in errors:
    tse_closed_form += error[0]**2
mse_closed_form = tse_closed_form/n

print('TSE Closed Form - Estimated a: ', tse_closed_form)
print('MSE Closed Form - Estimated a: ', mse_closed_form)

# Calculate for a equaling all zeroes.
a_zeros = np.zeros(100)
zero_errors = X.dot(a_zeros) - y
tse_zeros = 0
for error in zero_errors:
    tse_zeros += error[0]**2
mse_zeros = tse_zeros/n

print('TSE - a of zeros: ', tse_zeros)
print('MSE - a of zeros: ', mse_zeros)

#####################################################################################
# 1b: Gradient Decsent Solution
#####################################################################################

def gradient_descent(learning_rate, iterations, X = X, y = y, a_init = np.zeros((d, 1))):
    costs = []
    a = a_init

    for _ in range(iterations):
        y_pred = X.dot(a)
        cost = np.sum((y - y_pred)**2)
        costs.append(float(cost))

        # Calculate Gradients
        a_grad = -(X_t.dot(y - y_pred))

        # Update based on gradient.
        a = a - learning_rate * a_grad

    # Update cost one last time
    y_pred = X.dot(a)
    cost = np.sum((y - y_pred)**2)
    costs.append(float(cost))

    return a, costs

costs_dict = {}
learning_rates = [0.00005, 0.0005, 0.0007]
for learning_rate in learning_rates:
    coefficients, costs = gradient_descent(learning_rate, 20)
    if learning_rate == learning_rates[0]:
        print("Initital SSE: ", costs[0])
    print(f"Final SSE from Gradient Descent with learning rate of {learning_rate}:", costs[20])
    costs_dict[learning_rate] = costs

# Plotting the cost histories for different learning rates
plt.figure(figsize=(10, 6))

for learning_rate in learning_rates:
    plt.plot(costs_dict[learning_rate], label=f'Learning Rate: {learning_rate}')

# Adding labels and title
plt.xlabel('Iteration')
plt.ylabel('Cost (SSE)')
plt.title('SSE for Gradient Descent')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Given the three step sizes we tried here, the higher step sizes approached the minimum
# more quickly. Additionally, we did not see any jump around the the minimum from the 
# larger step sizes which would have been observed by the error increasing at certain
# steps. This would likely happen if we picked a step size much larger than the step 
# sizes chosen here.
    

#####################################################################################
# 1c: Stochastic Gradient Decsent Solution
#####################################################################################

def stochastic_gradient_descent(learning_rate, iterations, X = X, y = y, a_init = np.zeros((d, 1))):
    costs = []
    a = a_init # (100, 1)

    for _ in range(iterations):

        # Calculate Gradients
        selected = random.randint(0, len(y) - 1) # y is 1000 by 1
        x = X[selected].reshape(1, -1) # (1, 100)
        y_pred = np.dot(x, a) # (1, 1)
        a_grad = -(x * (y[selected] - y_pred)) # (1, 100)

        # Update based on gradient.
        a = a - learning_rate * a_grad.T

        # Update cost
        y_pred_total = X @ a
        cost = np.sum((y - y_pred_total)**2)
        costs.append(float(cost))

    return a, costs

costs_dict = {}
learning_rates = [0.0005, 0.005, 0.01]
for learning_rate in learning_rates:
    coefficients, costs = stochastic_gradient_descent(learning_rate, 1000)
    if learning_rate == learning_rates[0]:
        print("Initital SSE: ", costs[0])
    print(f"Final SSE from Stochastic Gradient Descent with learning rate of {learning_rate}:", costs[999])
    costs_dict[learning_rate] = costs

# Plotting the cost histories for different learning rates
plt.figure(figsize=(10, 6))

for learning_rate in learning_rates:
    plt.plot(costs_dict[learning_rate], label=f'Learning Rate: {learning_rate}')

# Adding labels and title
plt.xlabel('Iteration')
plt.ylabel('Cost (SSE)')
plt.title('SSE for Stochastic Gradient Descent')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# It appears the largest step size initially converges the fastest, but for most
# random environments the step size of 0.005 results in the smallest SSE of the
# step sizes tried here. The gradient descent algorithm uses each data point 20
# times (because we run 20 iterations), and the the stochastic gradient descent
# algorithm is expected to use each data point about 10 times (because we run 1000
# iterations.) Overall, in these environments, it seems the gradient descent 
# algorithm converges more quickly and more accurately in most of the environments
# tested here in part 1. 















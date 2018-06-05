#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 23:15:19 2018

@author: dileepn

Polynomial Regression: Predict office prices
"""
import numpy as np
import matplotlib.pyplot as plt

# Open the dataset and define X, y, and m
def load_data(filepath):
    
    # Open dataset with numpy to directly store it as a numpy array
    dataset = np.loadtxt(filepath, dtype=float)
    
    # Define X, y, and m
    n_features = (dataset.shape[1] - 1) # Assuming the last column is 'y'
    
    X = dataset[:, 0:n_features]
    y = dataset[:,dataset.shape[1] - 1].reshape(X.shape[0],1)
    
    m = dataset.shape[0]
    
    return (X, y, m)

# Feature scaling
def norm_features(X):
    
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    
    X_norm = (X - mu)/sigma
    
    # Add intercept term to X 
    X_norm = np.concatenate((np.ones((X_norm.shape[0], 1)), X_norm), axis=1)
    
    return (X_norm, mu, sigma)

# Polynomial feature mapping
def poly_features(X, p):
    
    # Define a new array to store poly features
    X_poly = np.zeros((X.shape[0], p))
    
    # Each column of X_poly is X raised to a power
    for i in range(0, p):
        X_poly[:, i] = X**(i + 1)
        
    return X_poly

# Compute cost and gradient
def compute_cost(X, y, theta, Lambda):
    
    # Calculate m
    m = y.shape[0]
    
    # Regularized cost
    J = (1/(2 * m)) * np.sum(np.square(X.dot(theta) - y.transpose())) + \
        (Lambda/(2 * m))*np.sum(np.square(theta[1:]));
        
    # Regularized gradient
    grad = (1/m) * X.transpose().dot((X.dot(theta) - y)) + \
            (Lambda/m) * np.concatenate((np.zeros((1,1)), theta[1:]), axis=0)
            
    return (J, grad)

# Gradient descent to learn theta
def grad_descent(X, y, theta, alpha, Lambda, num_iters):
    
    # Calculate m
    m = y.shape[0]
    
    # Array to hold cost, J for each iteration
    J_hist = np.zeros((num_iters, 1))
     
    # Start GD loop
    for i in range(num_iters):
        
        # Unpack grad and J
        J, grad = compute_cost(X, y, theta, Lambda)
        
        # Update theta
        theta = theta - alpha * (1/m) * grad
        
        # Store cost for iteration
        J_hist[i] = J
        
    # Return theta and J_hist
    return (theta, J_hist)

def train_cv_test_split(X, y, cv_ratio=0.2, test_ratio=0.2):
    
    # Randomly shuffle array indices before splitting data
    rand_indices = np.random.permutation(len(X))
    
    # Calculate CV and test set sizes
    cv_set_size = int(len(X) * cv_ratio)
    test_set_size = int(len(X) * test_ratio)
    
    # Get CV and test indices
    cv_indices = rand_indices[:cv_set_size]
    test_indices = rand_indices[cv_set_size:(cv_set_size + test_set_size)]
    train_indices = rand_indices[(cv_set_size + test_set_size):]
    
    # Split the dataset
    X_cv = X[cv_indices,:]
    y_cv = y[cv_indices,:]
    
    X_test = X[test_indices,:]
    y_test = y[test_indices,:]
    
    X_train = X[train_indices,:]
    y_train = y[train_indices,:]
    
    # Return split datasets
    return (X_cv, y_cv, X_test, y_test, X_train, y_train)

# Model training    

# Define constants and initial theta vector
filepath = "./input/input03.txt"
Lambda = 1
alpha = 0.1
num_iters = 1000

# Get X, y, and m from dataset
X, y, m = load_data(filepath)

# Normalize X to get mu = 0 and sigma = 1
X_norm, mu, sigma = norm_features(X)

# Learn theta using GD
theta_init = np.zeros((X_norm.shape[1], 1))
theta, J_hist = grad_descent(X_norm, y, theta_init, alpha, Lambda, num_iters)

# J_history plot
plt.figure(figsize=(7,7))
plt.title("Cost vs. iteration")
plt.xlabel("GD Iteration")
plt.ylabel("Cost, J")

plt.plot(range(1,num_iters+1), J_hist, 'b-', linewidth=2)

# Show plot
plt.show()

# Predictions
X_pred = np.array([[0.05, 0.54],[0.91, 0.91],[0.31, 0.76],[0.51, 0.31]])
X_pred = (X_pred - mu) / sigma
X_pred = np.concatenate((np.ones((X_pred.shape[0],1)), X_pred), axis=1)
predictions = X_pred.dot(theta)
print(predictions)


    
    





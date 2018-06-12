#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 23:15:19 2018

@author: dnackat

This script has functions to implement polynomial regression

"""
import numpy as np
import matplotlib.pyplot as plt

# Open the dataset and define X, y, and m
def load_data():
    """ This function loads: 
        1. Space-separated text (*.txt) file using numpy
        and returns X and y as numpy arrays. Sample input (last column is y):
            2 100
            0.44 0.68 511.14
            0.99 0.23 717.1
            0.84 0.29 607.91
            0.28 0.45 270.4
            0.07 0.83 289.88
                    .
                    .
                    .
            4
            0.05 0.54
            0.91 0.91
            0.31 0.76
            0.51 0.31
            
                    OR
        
        2. Data entered manually (space-separated) on the standard input 
        and stores them in X and y. """
    
    prompt_user = True
    while(prompt_user):
        
        # Prompt user for dataset input type
        input_type = input("Choose dataset input type. 1 (file) or 2 (manual entry): ")
    
        if input_type == "1":
            
            # Prompt for filepath
            filepath = input("Enter the complete filepath (/home/user...): ")
            
            # Temporary lists to store data as it is being read
            temp_data = []
            temp_test_data = []

            # Read the dataset line-by-line. Get num. of features, F, and 
            # num. of examples, N
            with open(filepath) as input_file:
            
                for line_num, line in enumerate(input_file):
                    if line_num == 0:
                        F, N = line.split()
                        F, N = int(F), int(N)
                    elif line_num == N + 1:
                        T = int(line)
                    elif line_num > 0 and line_num <= N:
                        x1, x2, y = line.split()
                        # Store as ordered pair in temp_data
                        temp_data += [(float(x1), float(x2), float(y))]
                    elif line_num > N + 1 and line_num <= N + T + 1:
                        x1, x2 = line.split()
                        temp_test_data += [(float(x1), float(x2))]
                    
            # Convert temp lists into numpy arrays
            dataset = np.array(temp_data)
            X_pred = np.array(temp_test_data)       
            
            # Define X, y, and m
            X = dataset[:, :F]
            y = dataset[:, F].reshape(X.shape[0], 1)
            m = dataset.shape[0]
            
            prompt_user = False
            
        elif input_type == "2":
            
            # First line has number of features and number of training examples
            F, N = map(int, input().split())
            
            # Get the training set (X and y)
            train = np.array([input().split() for _ in range(N)], float)
            
            # Number of test examples
            T = int(input())
            X_pred = np.array([input().split() for _ in range(T)], float)
            
            # Split the training set into X and y
            X = train[:,:F]
            y = train[:,F]
            m = len(y)
            
            prompt_user = False
        
        else:
            print("Incorrect input. Please enter 1 or 2.")
    
    return (X, y, m, X_pred)

# Feature scaling
def norm_features(X):
    """ This function normalizes features and returns X_norm: (X - mu)/sigma, 
        where mu and sigma are the mean and standard deviation of each column 
        of X. Also adds a column of 1's (intercept) to X"""
    
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    
    X_norm = (X - mu)/sigma
    
    # Add intercept term to X 
    X_norm = np.concatenate((np.ones((X_norm.shape[0], 1)), X_norm), axis=1)
    
    return X_norm

# Polynomial feature mapping
def poly_features(X, p):
    """ This function takes in X and returns a matrix X_poly
        with each column being a column of X raised to powers ranging from 
        1 through p. """
    
    # Get m 
    m = X.shape[0]
    
    # Define a new array to store poly features
    X_poly = np.zeros((m, (p**2 + 1)))
    
    # Each column of X_poly is X raised to a power
    index = 0
    for j in range(p + 1):
        for i in range(p + 1):
            if (i + j) <= p:
                X_poly[:, index] = (X[:, 0]**i) * (X[:, 1]**j)
                index += 1
    
    # Remove the first column with 1's before normalization
    X_poly = X_poly[:,1:]
    
    return X_poly

# Compute cost and gradient
def compute_cost(X, y, theta, Lambda):
    """ Compute regularized (if Lambda > 0) cost (J) and gradient. """
        
    # Calculate m
    m = y.shape[0]
    
    # Regularized cost
    J = (1/(2 * m)) * np.sum(np.square(X.dot(theta) - y)) + \
        (Lambda/(2 * m))*np.sum(np.square(theta[1:]));
        
    # Regularized gradient
    grad = (1/m) * X.transpose().dot((X.dot(theta) - y)) + \
            (Lambda/m) * np.concatenate((np.zeros((1,1)), theta[1:]), axis=0)
            
    return (J, grad)

# Gradient descent to learn theta
def grad_descent(X, y, theta, alpha, Lambda, num_iters):
    """ This function implements Gradient Descent to train the model, and 
        returns the tuned parameter vector, theta. """
    
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

# Split dataset into cv, test, and training sets
def train_cv_test_split(X, y, cv_ratio=0.2, test_ratio=0.2):
    """ This function splits the dataset into cross-validation, test, 
        and train sets. Default split is 20%-20%-60%. """
    
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

# Plot learning curve
def learning_curve(X_train, y_train, X_cv, y_cv, Lambda):
    """ This function plots the learning curve (train and validation errors
        vs. number of examples) to give an idea on bias-variance characteristics. """
    
    # Calculate m
    m = len(y_train)
    
    # Initialize error_train and error_cv vectors
    error_train = np.zeros((m, 1))
    error_cv = np.zeros((m, 1))
    
    # Start loop
    for i in range(m):
        
        # Calculate theta vector for this iteration
        theta_init = np.ones((X_train.shape[1], 1))
        theta, dummy_var = grad_descent(X_train[:i,:], y_train[:i], \
                                        theta_init, 0.1, Lambda, 500)
        
        # Calculate errors (set Lambda = 0 for these)
        error_train[i], dummy_var = compute_cost(X_train[:i,:], y_train[:i], \
                           theta, 0)
        error_cv[i], dummy_val = compute_cost(X_cv, y_cv, \
                           theta, 0)
        
    # Plot errors
    plt.figure(figsize=(7,7))
    plt.title("Learning curve for linear/poly regression")
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    
    plt.plot(range(m), error_train, 'b-', linewidth=2, label="Train")
    plt.plot(range(m), error_train, 'r-', linewidth=2, label="Cross Validation")
    
    plt.legend()
    plt.show()
    
    # Return error arrays
    return (error_train, error_cv)

# Plot data along with model fit
def plot_data_model(X, X_norm, y, theta):
    """ This function plots the model fit along with the data set (X vs. y). """
    
    plt.figure(figsize=(7,7))
    plt.title("Model fit")
    plt.xlabel("Feature (X)")
    plt.ylabel("Price per square-foot, y ($)")
    
    plt.plot(X[:,0], y, 'bx', markersize=5, label="x1")
    plt.plot(X[:,1], y, 'ro', markersize=5, label="x2")
    
    # Sort X in order to do a lineplot of model fit
    [x_p, y_p] = zip(*sorted(zip(X[:,0], X_norm.dot(theta)), \
    key=lambda x_p: x_p[0]))
    
    plt.plot(x_p, y_p, 'k--', linewidth=1, label="Model fit")
    
    plt.legend()
    plt.show()
    
    return
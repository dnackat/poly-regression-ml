#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 02:57:51 2018

@author: dnackat

This sript predicts office space prices based on 2 features using polynomial
regression.
 
"""
#%% Libraries
import numpy as np
import matplotlib.pyplot as plt

from polyRegression import load_data, norm_features, poly_features, compute_cost, \
grad_descent, train_cv_test_split, learning_curve, validationCurve, plot_data_model

# Model training and prediction   

#%% Define constants and initial theta vector
Lambda = 3
alpha = 0.125
num_iters = 3500

#%% Get X, y, and m from dataset
X, y, m, X_pred = load_data()

#%% Split set into train, cv, and test sets
X_cv, y_cv, X_test, y_test, X_train, y_train = train_cv_test_split(X, y)

#%% Get polynomial features
n_poly = 3
X_poly_train = poly_features(X_train, n_poly)
X_poly_cv = poly_features(X_cv, n_poly)
X_poly_test = poly_features(X_test, n_poly)

#%% Normalize X to get mu = 0 and sigma = 1
X_norm_train = norm_features(X_poly_train)
X_norm_cv = norm_features(X_poly_cv)
X_norm_test = norm_features(X_poly_test)

#%% Learn theta using GD
theta_init = np.random.randint(5, size=(X_norm_train.shape[1], 1))
theta, J_hist = grad_descent(X_norm_train, y_train, theta_init, alpha, Lambda, num_iters)

#%% J_history plot
plt.figure(figsize=(7,7))
plt.title("Cost vs. iteration")
plt.xlabel("GD Iteration")
plt.ylabel("Cost, J")

plt.plot(range(1,num_iters+1), J_hist, 'b-', linewidth=2)

plt.show()

#%% Plot fit
plot_data_model(X_train, X_norm_train, y_train, theta)

#%% Plot learning curve
error_train, error_cv = learning_curve(X_norm_train, y_train, \
                                        X_norm_cv, y_cv, Lambda)

#%% Plot validation curve
error_train_val, error_cv_val = validationCurve(X_norm_train, y_train, X_norm_cv, y_cv)

#%% Compute test set error
Lambda = 3
theta_test, _ = grad_descent(X_norm_train, y_train, theta_init, alpha, Lambda, num_iters)

# Calculate test set error
test_error, _ = compute_cost(X_norm_test, y_test, theta_test, 0)
print("--------------------------")
print("Test set error = {:.2f}".format(test_error))
print("--------------------------")

#%% Predictions
X_pred_poly = poly_features(X_pred, n_poly)
X_pred_norm = norm_features(X_pred_poly) 
predictions = X_pred_norm.dot(theta)

#%% Print predictions
print("--------------------------")
print("\nPredicted price(s) in $")
print("--------------------------")
for i in range(predictions.shape[0]):
    print("{:.2f}".format(float(predictions[i])))

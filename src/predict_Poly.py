#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 02:57:51 2018

@author: dileepn

This script predicts office space prices for the given test data
 
"""
import numpy as np
import matplotlib.pyplot as plt

from polyRegression import load_data, norm_features, poly_features, \
grad_descent, train_cv_test_split, learning_curve, plot_data_model

# Model training and prediction   

# Define constants and initial theta vector
Lambda = 0
alpha = 0.125
num_iters = 3500

# Get X, y, and m from dataset
X, y, m, X_pred = load_data()

# Get polynomial features
X_poly = poly_features(X, 3)

# Normalize X to get mu = 0 and sigma = 1
X_norm = norm_features(X_poly)

# Learn theta using GD
theta_init = np.random.randint(2, size=(X_norm.shape[1], 1))
theta, J_hist = grad_descent(X_norm, y, theta_init, alpha, Lambda, num_iters)

# J_history plot
plt.figure(figsize=(7,7))
plt.title("Cost vs. iteration")
plt.xlabel("GD Iteration")
plt.ylabel("Cost, J")

plt.plot(range(1,num_iters+1), J_hist, 'b-', linewidth=2)

plt.show()

# Plot fit
plot_data_model(X, X_norm, y, theta)

# Predictions
X_pred_poly = poly_features(X_pred, 3)
X_pred_norm = norm_features(X_pred_poly) 
predictions = X_pred_norm.dot(theta)

# Print predictions
print("\nPredicted price(s) in $")
print("--------------------------")
for i in range(predictions.shape[0]):
    print("{:.2f}".format(float(predictions[i])))

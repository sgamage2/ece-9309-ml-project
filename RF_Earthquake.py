# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:33:03 2019

@author: Bryan
"""
# In[1]:
import numpy as np
import pandas as pd
import time, os
import tensorflow as tf
import utility
import scipy.stats
import matplotlib.pyplot as plt
import statistics as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot as dot
import six
from sklearn import tree

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#from fastai.structured import draw_tree
# In[2]:
# Hardware (GPU or CPU)


def compute_statistical_features(window, moment_order):
    # num_moments = 1 + moment_order * 2  # mean + central moments, non-central moments
    mean = np.mean(window)
    moments = [mean]

    for n in range(2, moment_order+1):
        nth_central_moment = scipy.stats.moment(window, moment=n)
        nth_non_central_moment = np.mean(window ** n)
        moments.append(nth_central_moment)
        moments.append(nth_non_central_moment)

    return np.array(moments)


def prepare_dataset(dataset, window_size, window_stride):
    print("Preparing dataset")
    moment_order = 4
    series_length = dataset.shape[0]
    dataset = np.array(dataset)

    X_features = []
    y = []

    for i in range(window_size, series_length, window_stride):
        X_window = dataset[i - window_size:i, 0]
        features_of_window = compute_statistical_features(X_window, moment_order)
        X_features.append(features_of_window)

        y_of_window = dataset[i, 1]
        y.append(y_of_window)

    X_features, y = np.array(X_features), np.array(y)
    print("Preparing dataset complete")
    return X_features, y


def main():
    os.environ[
        'CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU as it appears to be slower than CPU (to enable GPU, comment out this line and restart the kernel)

    device_name = tf.test.gpu_device_name()

    if device_name:
        print('GPU device found: {}. Using GPU'.format(device_name))
    else:
        print("GPU device not found. Using CPU")
        # raise SystemError('GPU device not found')   # Stop the program if GPU is unavailabile: disabled for now

    # In[3]:
    # Variables that determines how the script behaves

    # Data was convereted from CSV to HDF then truncated
    hdf_key = 'my_key'

    # Change the following to point to proper local paths
    truncated_train_hdf_file = '../LANL-Earthquake-Prediction/truncated_train_hdf.h5'
    validation_hdf_file = '../LANL-Earthquake-Prediction/validation_hdf.h5'
    test_hdf_file = '../LANL-Earthquake-Prediction/test_hdf.h5'

    # Folder to save results
    results_dir = 'results/current_run'

    do_plot_series = False

    # In[4]:
    train_df = utility.read_hdf(truncated_train_hdf_file, hdf_key)
    valid_df = utility.read_hdf(validation_hdf_file, hdf_key)
    test_df = utility.read_hdf(test_hdf_file, hdf_key)

    # Parameters to tune
    down_sample = 5000
    window_size = 100
    window_stride = 20

    # Downsample the dataset
    train_df = train_df.iloc[::down_sample, :]
    valid_df = valid_df.iloc[::down_sample, :]
    test_df = test_df.iloc[::down_sample, :]

    # Derivative signals - does not give better results (slightly worse results)
    train_df = utility.derivative(train_df, 1)
    valid_df = utility.derivative(valid_df, 1)
    test_df = utility.derivative(test_df, 1)

    # Compute statistical features on a window
    Xtrain, Ytrain = prepare_dataset(train_df, window_size, window_stride)

    # Train
    print("Training Random Forest")
    reg = RandomForestRegressor(n_estimators=500, max_features=7, min_samples_split=30)
    reg.fit(Xtrain, Ytrain)

    print("Drawing forest")
    dotfile = six.StringIO()
    tree.export_graphviz(reg.estimators_[0], out_file='tree.dot', filled=True, rounded=True)
    os.system('dot -Tpng tree.dot -o tree.png')

    # Predict on training set
    print("Predicting on training set")
    Ytraing_pred = reg.predict(Xtrain)

    Ytraing_pred, Ytrain = utility.ma_filter(Ytraing_pred, Ytrain, 25)

    train_mse, train_rmse, train_mae, train_r2 = utility.metrics(Ytrain, Ytraing_pred)

    print(
        'Error metrics on training set. train_mse: {:.4f}, train_rmse: {:.4f}, train_mae: {:.4f}, train_r2: {:.4f}'.format(
            train_mse, train_rmse, train_mae, train_r2))

    plt.figure()
    plt.plot(Ytrain)
    plt.plot(Ytraing_pred)

    # plt.show()

    # Predict on validation set

    Xvalid, Yvalid = prepare_dataset(valid_df, window_size, window_stride)

    print("Predicting on validation set")
    Yvalid_pred = reg.predict(Xvalid)

    Yvalid_pred, Yvalid = utility.ma_filter(Yvalid_pred, Yvalid, 25)

    valid_mse, valid_rmse, valid_mae, valid_r2 = utility.metrics(Yvalid, Yvalid_pred)

    print(
        'Error metrics on validation set. validation_mse: {:.4f}, validation_rmse: {:.4f}, validation_mae: {:.4f}, validation_r2: {:.4f}'.
        format(valid_mse, valid_rmse, valid_mae, valid_r2))

    plt.figure()
    plt.plot(Yvalid)
    plt.plot(Yvalid_pred)

    # Predict on test set

    Xtest, Ytest = prepare_dataset(test_df, window_size, window_stride)

    print("Predicting on test set")
    Ytest_pred = reg.predict(Xtest)

    Ytest_pred, Ytest = utility.ma_filter(Ytest_pred, Ytest, 25)

    test_mse, test_rmse, test_mae, test_r2 = utility.metrics(Ytest, Ytest_pred)

    print('Error metrics on test set. test_mse: {:.4f}, test_mse: {:.4f}, test_mae: {:.4f}, test_r2: {:.4f}'.format(
        test_mse, test_rmse, test_mae, test_r2))

    plt.figure()
    plt.plot(Ytest)
    plt.plot(Ytest_pred)

    plt.show()


if __name__ == "__main__":
    main()


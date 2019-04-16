# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:33:03 2019

@author: Bryan
"""
# In[1]:
import numpy as np
import os, time
import utility
import scipy.stats

import matplotlib
matplotlib.use('Agg')

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot as dot
import six
from sklearn import tree
import pickle

# Variables that determines how the script behaves

# Data was converted from CSV to HDF then truncated
hdf_key = 'my_key'

# Change the following to point to proper local paths
truncated_train_hdf_file = '../LANL-Earthquake-Prediction/truncated_train_hdf.h5'
validation_hdf_file = '../LANL-Earthquake-Prediction/validation_hdf.h5'
test_hdf_file = '../LANL-Earthquake-Prediction/test_hdf.h5'

# Folder to save results
results_dir = 'results/random_forest_run'

# Parameters to tune
down_sample = 1000
window_size = 100
window_stride = 20
num_trees = 1000

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


def do_predictions(model, X, y, dataset_name, filename):
    y_pred = model.predict(X)

    y_pred, y = utility.ma_filter(y_pred, y, 25)

    mse, rmse, mae, r2 = utility.metrics(y, y_pred)

    print('Error metrics on {}. mse: {:.4f}, rmse: {:.4f}, mae: {:.4f}, r2: {:.4f}'.
            format(dataset_name, mse, rmse, mae, r2))

    title = 'True vs predicted time_to_earthquake on {}'.format(dataset_name)

    utility.plot_results(y, y_pred, title, filename)


def main():
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print('\n================= Random Forest parameters used for this run ================= \n')
    print("down_sample = {} \nwindow_size = {} \nwindow_stride = {} \nnum_trees = {}"
          .format(down_sample, window_size, window_stride, num_trees, ))
    print('\n============================================================================== \n')

    train_df = utility.read_hdf(truncated_train_hdf_file, hdf_key)
    valid_df = utility.read_hdf(validation_hdf_file, hdf_key)
    test_df = utility.read_hdf(test_hdf_file, hdf_key)

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
    Xvalid, Yvalid = prepare_dataset(valid_df, window_size, window_stride)
    Xtest, Ytest = prepare_dataset(test_df, window_size, window_stride)

    print("Dataset sizes. Xtrain={}, Xvalid={}, Xtest={}".format(Xtrain.shape, Xvalid.shape, Xtest.shape))

    # Train the model
    print("Training Random Forest")
    t0 = time.time()
    reg = RandomForestRegressor(n_estimators=num_trees, max_features=7, min_samples_split=30)
    reg.fit(Xtrain, Ytrain)
    time_to_train = time.time() - t0
    print("Training complete. time_to_train = {} seconds".format(time_to_train))

    print("Drawing forest")
    dotfile = six.StringIO()
    tree.export_graphviz(reg.estimators_[0], out_file='tree.dot', filled=True, rounded=True)
    os.system('dot -Tpng tree.dot -o tree.png')

    # Save model
    model_filename = results_dir + '/' + 'rf_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(reg, file)

    # Predict on training set
    print("Predicting on training set")
    t0 = time.time()
    train_res_plot_filename = results_dir + '/' + 'train_true_vs_pred' + '.png'
    do_predictions(reg, Xtrain, Ytrain, "training set", train_res_plot_filename)
    print("Predictions on training set complete. time_to_predict = {}".format(time.time() - t0))

    # Predict on validation set
    print("Predicting on validation set")
    t0 = time.time()
    valid_res_plot_filename = results_dir + '/' + 'validation_true_vs_pred' + '.png'
    do_predictions(reg, Xvalid, Yvalid, "validation set", valid_res_plot_filename)
    print("Predictions on validation set complete. time_to_predict = {}".format(time.time() - t0))

    # Predict on test set
    print("Predicting on test set")
    t0 = time.time()
    test_res_plot_filename = results_dir + '/' + 'test_true_vs_pred' + '.png'
    do_predictions(reg, Xtest, Ytest, "test set", test_res_plot_filename)
    print("Predictions on test set complete. time_to_predict = {}".format(time.time() - t0))


if __name__ == "__main__":
    main()


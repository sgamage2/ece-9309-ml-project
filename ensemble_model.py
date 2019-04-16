import argparse, os
import pickle
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import utility
import RF_Earthquake as rf

# ------------------------------------------------
# Variables that determines how the script behaves

# Data was converted from CSV to HDF then truncated
hdf_key = 'my_key'

# Change the following to point to proper local paths
truncated_train_hdf_file = '../LANL-Earthquake-Prediction/truncated_train_hdf.h5'
validation_hdf_file = '../LANL-Earthquake-Prediction/validation_hdf.h5'
test_hdf_file = '../LANL-Earthquake-Prediction/test_hdf.h5'

# Folder to save results
results_dir = 'results/ensemble'

# Common params
down_sample = 1000

# LSTM params
scaling_type = 'None'
features = ['original', 'derivative_1', 'derivative_2']
time_steps = 200
window_stride = 40

# RF params
window_size = 100
# ------------------------------------------------


def get_args():
    parser = argparse.ArgumentParser(description='Ensemble model')

    parser.add_argument("--lstm_model_file", required=True, type=str)
    parser.add_argument("--rf_model_file", required=True, type=str)

    args = parser.parse_args()

    return args.lstm_model_file, args.rf_model_file


def load_models(lstm_model_file, rf_model_file):
    lstm_model = load_model(lstm_model_file)
    print("Loaded Keras LSTM model")
    lstm_model.summary()

    with open(rf_model_file, 'rb') as file:
        rf_model = pickle.load(file)
    print("Loaded Random Forest model")

    return lstm_model, rf_model


def lstm_do_scaling(train_df, valid_df, test_df, scaling_type):
    train_df = train_df.values
    valid_df = valid_df.values
    test_df = test_df.values

    if scaling_type == 'None':
        training_set_scaled = train_df
        valid_set_scaled = valid_df
        test_set_scaled = test_df
    else:
        if scaling_type == 'MinMaxScaler':
            scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaling_type == 'StandardScaler':
            scaler = StandardScaler()

        signal_scaled = scaler.fit_transform(train_df[:, 0].reshape(-1, 1))
        training_set_scaled = train_df.copy()  # May not be necessary
        training_set_scaled[:, 0] = signal_scaled.reshape(-1)

        signal_scaled = scaler.fit_transform(valid_df[:, 0].reshape(-1, 1))
        valid_set_scaled = valid_df.copy()  # May not be necessary
        valid_set_scaled[:, 0] = signal_scaled.reshape(-1)

        signal_scaled = scaler.fit_transform(test_df[:, 0].reshape(-1, 1))
        test_set_scaled = test_df.copy()  # May not be necessary
        test_set_scaled[:, 0] = signal_scaled.reshape(-1)

    return training_set_scaled, valid_set_scaled, test_set_scaled


def lstm_engineer_features(training_set_scaled, valid_set_scaled, test_set_scaled, features):
    train_features = []
    train_ys = []

    valid_features = []
    valid_ys = []

    test_features = []
    test_ys = []

    for feature in features:
        train_feature = utility.get_feature(training_set_scaled, feature)
        train_features.append(train_feature.iloc[:, 0])  # 0th column contains the feature
        train_ys.append(train_feature.iloc[:, 1])  # 1st column contains y

        valid_feature = utility.get_feature(valid_set_scaled, feature)
        valid_features.append(valid_feature.iloc[:, 0])
        valid_ys.append(valid_feature.iloc[:, 1])

        test_feature = utility.get_feature(test_set_scaled, feature)
        test_features.append(test_feature.iloc[:, 0])
        test_ys.append(test_feature.iloc[:, 1])

    utility.trim_to_same_length(train_features, train_ys)
    utility.trim_to_same_length(valid_features, valid_ys)
    utility.trim_to_same_length(test_features, test_ys)

    train_ys = train_ys[0]  # Because all the ys dataframes in this list are the same
    valid_ys = valid_ys[0]
    test_ys = test_ys[0]

    return (train_features, train_ys), (valid_features, valid_ys), (test_features, test_ys)


def lstm_create_windows(data_features, ys, time_steps, window_stride, num_features):
    X_data = []
    y_data = []

    for i in range(time_steps, ys.shape[0], window_stride):
        X_window = np.zeros((time_steps, num_features))
        for j, feature_df in enumerate(data_features):
            X_window[:, j] = feature_df.iloc[i - time_steps:i]

        X_data.append(X_window)
        y_data.append(ys.iloc[i])

    X_data = np.array(X_data)
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], num_features))
    y_data = np.reshape(y_data, (-1, 1))  # Not required when we are predicting just one output value, but necessary when we predict more

    print(X_data.shape)
    print(y_data.shape)

    return X_data, y_data


def prepare_datasets_for_lstm(train_df, valid_df, test_df):
    training_set_scaled, valid_set_scaled, test_set_scaled = lstm_do_scaling(train_df, valid_df, test_df, scaling_type)

    (train_features, train_ys), (valid_features, valid_ys), (test_features, test_ys) = lstm_engineer_features(training_set_scaled, valid_set_scaled, test_set_scaled, features)

    num_features = len(features)

    X_train, y_train = lstm_create_windows(train_features, train_ys, time_steps, window_stride, num_features)
    X_valid, y_valid = lstm_create_windows(valid_features, valid_ys, time_steps, window_stride, num_features)
    X_test, y_test = lstm_create_windows(test_features, test_ys, time_steps, window_stride, num_features)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def prepare_datasets_for_rf(train_df, valid_df, test_df):
    train_df = utility.derivative(train_df, 1)
    valid_df = utility.derivative(valid_df, 1)
    test_df = utility.derivative(test_df, 1)

    Xtrain, Ytrain = rf.prepare_dataset(train_df, window_size, window_stride)
    Xvalid, Yvalid = rf.prepare_dataset(valid_df, window_size, window_stride)
    Xtest, Ytest = rf.prepare_dataset(test_df, window_size, window_stride)

    return (Xtrain, Ytrain), (Xvalid, Yvalid), (Xtest, Ytest)

def plot_results(true_series, lstm_pred_series, rf_pred_series, ensemble_pred_series, title, filename):
    plt.figure()
    plt.plot(true_series, color = 'red', label = 'True time_to_earthquake')
    plt.plot(lstm_pred_series, color = 'blue', label = 'Predictions from LSTM')
    plt.plot(rf_pred_series, color = 'green', label = 'Predictions from RF')
    plt.plot(ensemble_pred_series, color = 'magenta', label = 'Predictions from Ensemble')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('time_to_earthquake')
    plt.legend()

    plt.savefig(filename, bbox_inches='tight')
    print('Predictions plot saved to: {}'.format(filename))


def compare_models(lstm_model, rf_model, X_lstm, y_lstm, X_rf, y_rf, dataset_name, filename):
    y_pred_lstm = lstm_model.predict(X_lstm)
    y_pred_lstm, y_lstm = utility.ma_filter(y_pred_lstm, y_lstm, 25)

    mse, rmse, mae, r2 = utility.metrics(y_lstm, y_pred_lstm)
    print('LSTM error metrics on {}. mse: {:.4f}, rmse: {:.4f}, mae: {:.4f}, r2: {:.4f}'.
          format(dataset_name, mse, rmse, mae, r2))

    y_pred_rf = rf_model.predict(X_rf)
    y_pred_rf, y_rf = utility.ma_filter(y_pred_rf, y_rf, 25)

    mse, rmse, mae, r2 = utility.metrics(y_rf, y_pred_rf)
    print('RF error metrics on {}. mse: {:.4f}, rmse: {:.4f}, mae: {:.4f}, r2: {:.4f}'.
          format(dataset_name, mse, rmse, mae, r2))

    y_pred_ensemble = ensemble_predict(y_pred_lstm, y_pred_rf)

    mse, rmse, mae, r2 = utility.metrics(y_rf, y_pred_ensemble)
    print('Ensemble error metrics on {}. mse: {:.4f}, rmse: {:.4f}, mae: {:.4f}, r2: {:.4f}'.
          format(dataset_name, mse, rmse, mae, r2))

    title = 'True vs predicted time_to_earthquake on {}'.format(dataset_name)
    plot_results(y_rf, y_pred_lstm, y_pred_rf, y_pred_ensemble, title, filename)


def trim_from_start(X1, y1, X2, y2):
    n1 = len(X1)
    n2 = len(X2)

    if n1 > n2:
        N = n1 - n2
        X1 = X1[N:, :]
        y1 = y1[N:]
    elif n2 > n1:
        N = n2 - n1
        X2 = X2[N:, :]
        y2 = y2[N:]

    return (X1, y1), (X2, y2)


def ensemble_predict(lstm_predict, rf_predict):
    return (lstm_predict + rf_predict) / 2


"""
Load the 2 models (LSTM and Random Forest), and make predictions on the 3 datasets
"""
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    lstm_model_file, rf_model_file = get_args()

    lstm_model, rf_model = load_models(lstm_model_file, rf_model_file)

    train_df = utility.read_hdf(truncated_train_hdf_file, hdf_key)
    valid_df = utility.read_hdf(validation_hdf_file, hdf_key)
    test_df = utility.read_hdf(test_hdf_file, hdf_key)

    # Downsample the dataset
    train_df = train_df.iloc[::down_sample, :]
    valid_df = valid_df.iloc[::down_sample, :]
    test_df = test_df.iloc[::down_sample, :]

    (X_train_lstm, y_train_lstm), (X_valid_lstm, y_valid_lstm), (X_test_lstm, y_test_lstm) = prepare_datasets_for_lstm(train_df, valid_df, test_df)

    (X_train_rf, y_train_rf), (X_valid_rf, y_valid_rf), (X_test_rf, y_test_rf) = prepare_datasets_for_rf(train_df, valid_df, test_df)

    (X_train_lstm, y_train_lstm), (X_train_rf, y_train_rf) = trim_from_start(X_train_lstm, y_train_lstm, X_train_rf, y_train_rf)
    (X_valid_lstm, y_valid_lstm), (X_valid_rf, y_valid_rf) = trim_from_start(X_valid_lstm, y_valid_lstm, X_valid_rf, y_valid_rf)
    (X_test_lstm, y_test_lstm), (X_test_rf, y_test_rf) = trim_from_start(X_test_lstm, y_test_lstm, X_test_rf, y_test_rf)
    print('X_train_rf.shape = {}, y_train_rf.shape = {}'.format(X_train_rf.shape, y_train_rf.shape))

    train_res_plot_filename = results_dir + '/' + 'train_true_vs_pred' + '.png'
    compare_models(lstm_model, rf_model, X_train_lstm, y_train_lstm, X_train_rf, y_train_rf, "training set", train_res_plot_filename)

    valid_res_plot_filename = results_dir + '/' + 'validation_true_vs_pred' + '.png'
    compare_models(lstm_model, rf_model, X_valid_lstm, y_valid_lstm, X_valid_rf, y_valid_rf, "validation set", valid_res_plot_filename)

    test_res_plot_filename = results_dir + '/' + 'test_true_vs_pred' + '.png'
    compare_models(lstm_model, rf_model, X_test_lstm, y_test_lstm, X_test_rf, y_test_rf, "test set", test_res_plot_filename)


if __name__ == "__main__":
    main()

import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def read_hdf(filename, key):
    t0 = time.time()
    print('Reading HDF dataset {}'.format(filename))

    dataset_df = pd.read_hdf(filename, key=key)

    print('Reading complete. time_to_read={:.2f} seconds'.format(time.time() - t0))

    return dataset_df  # This is a Pandas DataFrame


def read_csv(filename):
    t0 = time.time()
    print('Reading CSV dataset {}'.format(filename))

    dataset_df = pd.read_csv(filename, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

    print('Reading {} complete. time_to_read={}'.format(filename, time.time() - t0))

    return dataset_df   # This is a Pandas DataFrame


def write_to_hdf(df, filename, key, compression_level):
    print('Writing dataset to HDF5 format. filename={}'.format(filename))
    t0 = time.time()

    df.to_hdf(filename, key=key, mode='w', complevel=compression_level, complib='zlib')

    print('Writing complete. time_to_write={}'.format(time.time() - t0))


def print_info(df):
    print("Dataset shape = {}".format(df.shape))
    df.info(memory_usage='deep')    # Some info about the dataset (memory usage etc.)

    pd.set_option("display.precision", 15)  # Show more decimals
    print(df.head())

    #print(df.acoustic_data.describe())  # Some stats; can take time if dataset is large
    #print(df.time_to_failure.describe())


def truncate_dataset(df, start_row, end_row):
    print('Truncating the dataset. start_row={}, end_row={}'.format(start_row, end_row))
    trunc_df = df[start_row:end_row]
    print('Truncating complete. trunc_df.shape={}'.format(trunc_df.shape))
    return trunc_df


def plot_series(df, title, save_dir):
    print('Plotting series')
    t0 = time.time()
    
    fig, ax1 = plt.subplots()
    plt.title(title)

    color = 'tab:orange'
    ax1.set_xlabel('time (#points)')
    ax1.set_ylabel('seismic_signal', color=color)
    ax1.plot(df.acoustic_data, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('time_to_earthquake', color=color)  # we already handled the x-label with ax1
    ax2.plot(df.time_to_failure, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    filename = save_dir + '/' + title + '.png'
    plt.savefig(filename, bbox_inches='tight')
    print('Plot saved to: {}'.format(filename))
    #plt.show()
    
    print('Plotting complete. time_to_plot={:.2f} seconds'.format(time.time() - t0))


def plot_training_history(history, save_dir):
    plt.figure()
    plt.title("Training history")
    
    plt.plot(history.history['loss'], label='training_loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='validation_loss')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    
    filename = save_dir + '/' + 'training_history' + '.png'
    plt.savefig(filename, bbox_inches='tight')
    print('Plot saved to: {}'.format(filename))


def plot_results(true_series, pred_series, title, filename):
    plt.figure()
    plt.plot(true_series, color = 'red', label = 'True time_to_earthquake')
    plt.plot(pred_series, color = 'blue', label = 'Predicted time_to_earthquake')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('time_to_earthquake')
    plt.legend()

    plt.savefig(filename, bbox_inches='tight')
    print('Predictions plot saved to: {}'.format(filename))


def set_params_from_command_line(params):
    parser = argparse.ArgumentParser(description='LSTM notebook')

    for key, val in params.__dict__.items():
        switch_name = "--" + key
        param_type = type(val)
        if param_type == list:
            param_type = str

        # print(switch_name)
        parser.add_argument(switch_name, required=False, type=param_type)

    args = parser.parse_args()

    for key, val in params.__dict__.items():
        param_type = type(val)
        arg_param_value = getattr(args, key)

        if param_type == list:
            element_type = type(val[0])
            arg_param_value = [element_type(item) for item in arg_param_value.split(',')]

        setattr(params, key, arg_param_value)


def print_params(params):
    for key, val in params.__dict__.items():
        print('{} = {}'.format(key, val))
        param_type = type(val)


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, rmse, mae, r2


def ma_filter(y1, y2, N):
    y1 = y1.reshape((-1,))  # For pandas MA function below
    y1 = pd.Series(y1).rolling(window=N).mean().iloc[N - 1:].values
    y2 = y2[N - 1:]  # Because the MA filtered version is shorter than the original
    return y1, y2


def derivative(dataset, order):
    dataset = np.array(dataset)
    X, y = dataset[:, 0], dataset[:, -1]
    X_diff = np.diff(X, n=order)
    y_shortened = y[order:] # Drop some values from the beginning
    assert len(X_diff) == len(y_shortened)
    data = {'X_diff': X_diff, 'y_shortened': y_shortened}
    dataset_diff = pd.DataFrame(data)
    return dataset_diff


def get_feature(dataset, feature):
    if feature == 'original':
        data = {'X': dataset[:, 0], 'y': dataset[:, 1]}
        original = pd.DataFrame(data)
        return original
    elif feature == 'log':
        assert False    # Cannot support log, as the signal has negative values
        x_log = np.log(dataset[:, 0])
        data = {'X': x_log, 'y': dataset[:, 1]}
        logarithm = pd.DataFrame(data)
        return logarithm
    elif feature.startswith("derivative"):
        elements = feature.split("_")
        assert len(elements) == 2
        order = int(elements[1])
        dataset_diff = derivative(dataset, order)
        return dataset_diff
    else:
        assert False, 'Unsupported feature: {}'.format(feature)


def trim_to_same_length(features, ys):
    lengths = []
    for feature_df in features:
        lengths.append(feature_df.shape[0])

    min_length = min(lengths)

    for feature_df, y_df in zip(features, ys):
        cur_len = feature_df.shape[0]
        to_drop = cur_len - min_length
        if to_drop > 0:
            # print("Dropping, cur_len={}, min_length={}".format(cur_len, min_length))
            feature_df.drop(feature_df.head(to_drop).index, inplace=True)
            y_df.drop(y_df.head(to_drop).index, inplace=True)

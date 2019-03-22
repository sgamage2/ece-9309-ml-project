import numpy as np
import pandas as pd

import time
import os

import matplotlib.pyplot as plt

import utility


# --------------------- Configurations ---------

# Input files (csv/ hdf5)
train_csv_file = '../LANL-Earthquake-Prediction/train.csv'
test_files_directory = '../LANL-Earthquake-Prediction/test'
test_file_1 = '../LANL-Earthquake-Prediction/test/seg_00a37e.csv'
train_hdf_file = '../LANL-Earthquake-Prediction/train_hdf_converted.h5'
hdf_key = 'my_key'

# Training dataset parameters
train_start_row = 0
train_end_row = 139000000    # 22% out of 620 million rows (captures 4 earthquakes)
truncated_train_hdf_file = '../LANL-Earthquake-Prediction/truncated_train_hdf.h5'

# Validation dataset parameters
validation_start_row = 139000000
validation_end_row = 219000000    # 13% out of 620 million rows (captures 2 earthquakes)
validation_hdf_file = '../LANL-Earthquake-Prediction/validation_hdf.h5'

# Test dataset parameters
test_start_row = 219000000
test_end_row = 246000000    # 4% out of 620 million rows (captures 1 earthquake)
test_hdf_file = '../LANL-Earthquake-Prediction/test_hdf_xxxxx.h5'

# -----------------------------------------------


'''
This function will read the given (CSV) dataset, convert and write it to a HDF file, 
truncate the dataset, and write the truncated dataset to a HDF file
Run it only once
'''
def prepare_training_dataset():
    train_df = utility.read_csv(train_csv_file)  # Only need to read from once (~9 minutes). Then write to an HDF file and use that afterwards
    # train_df = read_hdf(train_hdf_file, hdf_key)
    utility.print_info(train_df)

    utility.write_to_hdf(train_df, train_hdf_file, hdf_key, 5)  # Only need to write once

    trunc_train_df = utility.truncate_dataset(train_df, train_start_row, train_end_row)

    utility.write_to_hdf(trunc_train_df, truncated_train_hdf_file, hdf_key, 5)

'''
Load full dataset from HDF file, extract data points between given indexes and write to a new HDF file
'''
def prepare_validation_dataset():
    validation_df = utility.read_hdf(train_hdf_file, hdf_key)
    trunc_validation_df = utility.truncate_dataset(validation_df, validation_start_row, validation_end_row)
    utility.write_to_hdf(trunc_validation_df, validation_hdf_file, hdf_key, 5)


'''
This is similar to the function prepare_validation_dataset
'''
def prepare_test_dataset():
    test_df = utility.read_hdf(train_hdf_file, hdf_key)
    trunc_train_df = utility.truncate_dataset(test_df, test_start_row, test_end_row)
    utility.write_to_hdf(trunc_train_df, test_hdf_file, hdf_key, 5)


def file_len(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def file_lengths_info(directory):
    print('Checking lengths of files in directory: {}'.format(directory))
    lengths = []

    for f in os.listdir(directory):
        full_filename = os.path.join(directory, f)
        if os.path.isfile(full_filename):
            l = file_len(full_filename)
            lengths.append(l)

    max_length = max(lengths)
    min_length = min(lengths)
    avg_length = sum(lengths) / len(lengths)

    print('Checking lengths complete')
    print('min_length={}, max_length={}, avg_length={}'.format(min_length, max_length, avg_length))



# -----------------------------------------------


# ----- Script starts running from here ---------

if __name__ == "__main__":
    # Following 3 function calls are meant to be run only once
    # prepare_training_dataset()
    # prepare_validation_dataset()
    # prepare_test_dataset()

    # file_lengths_info(test_files_directory) # Just to check the signal size in test files

    # # ---- We will now work with the truncated training set
    #
    train_df = utility.read_hdf(train_hdf_file, hdf_key)
    utility.print_info(train_df)
    
    #utility.plot_series(train_df, 'Truncated_training_series', './')
    
    #plt.show()

    pass

# -----------------------------------------------





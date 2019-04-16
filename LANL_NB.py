
# coding: utf-8

# # LANL Notebook

# Link to competition and dataset: https://www.kaggle.com/c/LANL-Earthquake-Prediction
# 
# Anaysis: https://www.kaggle.com/jsaguiar/seismic-data-exploration
# 
# 
# ## The Dataset
# The training dataset has 2 columns (x, y), representing a single, continuous seismic signal.  x is the signal value, and y is the time-to-next-earthquake. The test data consists of segments of seismic signals (x values), and for each signal point in a segment, a time-to-next-earthquake (y) must be predicted by a machine learning model.
# 
# The dataset is large with over 600 million data points in the training signal (9 GB in csv format) and working with it will prove to be challenging (loading, visualizing, training models etc.).
# 

# ## Useful Ideas
# 
# 1. Downsampling to reduce dataset size:
# > There are ways to do this without aliasing error (or minimal aliasing error). See the [Wiki article](https://en.wikipedia.org/wiki/Downsampling_(signal_processing)) (short).
# 
# 
# 2. Noise filtering and other signal enhancements:
# > The signal *may* be noisy. Any good filters (especially those used in the seismic domain) will improve our model accuracy.
# 
# 
# 3. Feature engineering: the raw signal maybe 'too raw' for the RNN to learn useful/ predictive patterns. We may have to engineer features.
# 
#     3.1. Feature detecting filters
# > Features in signals are usually detected by filters (a filter is defined by its kernel: the impulse response). There may be feature detecting filters already used in the seismic analysis domain. Eg: filters to detect the presence of a peak.
# 
#     3.2. Engineering other features
# > We can engineering other features such as: does_peak_exist_in_this_window, time_since_last_peak, or other signals derived from the original such as first_derivative, second_derivative, Fourier transform (DFT) of the window, moving_average_smoothed, etc.
#     
#     3.3. Map data to higher dimension, e.g. using Kernels
# 
# 
# 4. Automating feature engineering via convolutional nets (CNN)
# > CNNs essentially learn the kernels of filters as part of the neural network. We can have some CNN layers before the LSTM layers and see if that works.
# 
# 
# 5. I checked some of the test files and it seems that the peaks are absent. Hence, I stongly recommend completely eliminating the peaks before training the RNN. Peaks will be considered outliers.
# 
# 
# 6. On the side, we will briefly try a different model, at least to answer the question "Did you try any other models?"

# In[1]:


# Imports for the script

import numpy as np
import pandas as pd
import sys
import time, os

import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
#%matplotlib inline

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

# To visualize the RNN network
from keras.utils.vis_utils import plot_model
import pydot

import utility  # Contains various helper utility functions


# In[2]:


# Hardware (GPU or CPU)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # Disable GPU as it appears to be slower than CPU (to enable GPU, comment out this line and restart the kernel)

device_name = tf.test.gpu_device_name()

if device_name:
    print('GPU device found: {}. Using GPU'.format(device_name))
else:
    print("GPU device not found. Using CPU")
    #raise SystemError('GPU device not found')   # Stop the program if GPU is unavailabile: disabled for now


# In[3]:


# Variables that determines how the script behaves

# Data was convereted from CSV to HDF then truncated
hdf_key = 'my_key'

# Change the following to point to proper local paths
truncated_train_hdf_file = '../LANL-Earthquake-Prediction/truncated_train_hdf.h5' 
validation_hdf_file = '../LANL-Earthquake-Prediction/validation_hdf.h5'
test_hdf_file = '../LANL-Earthquake-Prediction/test_hdf.h5'

do_plot_series = False   # Whether training, validation, testing series should be plotted (time and memory consuming: keep this disabled unless necessary)


# In[4]:


# Tunable parameters relating to the operation of the algorithm

# Dummy class to hold parameters
class Parameters():
    pass

params = Parameters()

# Data preprocessing
params.scaling_type = 'None'   # Supports: None, StandardScaler, MinMaxScaler
params.down_sample = 40000
params.features = ['original', 'derivative_1', 'derivative_2']   # Currently supports: original, derivative_n

# LSTM network architecture
params.time_steps = 50
params.window_stride = 25
params.rnn_layer_units = [10, 5, 2]   # The length of this list = no. of hidden layers
params.rnn_layer_dropout_rate = [0.2, 0.2, 0]   # Dropout rate for each layer (0 for no dropout)

# Training
params.epochs = 2
params.batch_size = 32

# Post-processing
params.ma_window = 25   # The size of the Moving Average filter that will be applied to the output time_to_earthquake

# Admin variables unrelated to RNN parameters
params.results_dir = 'results/current_run'   # Folder to save results
params.experiment_num = -1   # For bookkeeping; currently not used
params.description = ''   # To print in the log


# In[5]:


# If certain params are provided as command line arguments to this notebook, they are set here, overriding the values set in the code above
# This is done so that this notebook can be run from the command line or from another script, thereby automating the experimentation process

if '-f' in sys.argv: # To prevent error given when running this cell on the Jupyter notebook
    sys.argv.remove('-f')

if len(sys.argv) > 2:   # Command line args available
    utility.set_params_from_command_line(params)


# In[6]:


print('\n================= RNN parameters used for this run ================= \n')
utility.print_params(params)
print('\n==================================================================== \n')


# In[7]:


# Some checks to ensure the parameters are valid
assert len(params.rnn_layer_units) == len(params.rnn_layer_dropout_rate)
assert params.scaling_type == 'None' or params.scaling_type == 'StandardScaler' or params.scaling_type == 'MinMaxScaler'


# In[8]:


if not os.path.exists(params.results_dir):
    os.makedirs(params.results_dir)


# ## Import truncated Data from hdf files
# ### 3 sets: train, validation, test. Print basic stats about them and plot them

# In[9]:


train_df = utility.read_hdf(truncated_train_hdf_file, hdf_key)
valid_df = utility.read_hdf(validation_hdf_file, hdf_key)
test_df = utility.read_hdf(test_hdf_file, hdf_key)


# In[10]:


utility.print_info(train_df)

if do_plot_series:
    utility.plot_series(train_df, "Training series", params.results_dir) # This is time and memory consuming. Do not run this cell unless necessary


# In[11]:


utility.print_info(valid_df)

if do_plot_series:
    utility.plot_series(valid_df, "Validation series", params.results_dir) # This is time and memory consuming. Do not run this cell unless necessary


# In[12]:


utility.print_info(test_df)

if do_plot_series:
    utility.plot_series(test_df, "Testing series", params.results_dir) # This is time and memory consuming. Do not run this cell unless necessary


# # Understanding Data
# 
# ## Domain Perspective
# 
# https://www.youtube.com/watch?v=T0AEtX-uPLA
# - Earthquakes occur when two parts of the earth suddenly move in relation to each other
# - Earthequake originates at a point in earth called focus and spreads up to the surface at a point called Epicenter.
# - A seismograph detects and records seismic waves. \n",
# - Types of Seismic waves include:  Pressure (or Primary) Waves (P-Waves), Shear (or Secondary) Waves (S-Waves) and Surface Waves. P-Waves and S-Waves are both Body waves.
# 
# https://courses.lumenlearning.com/geophysical/chapter/earthquake-prediction/
# - Predciting when an earthquake will occur is more difficult than predicting where it will occur.
# - Sometimes (not always) earthquakes occur few seconds to few weeks after foreshocks.
# 
# https://en.wikipedia.org/wiki/P-wave
# - P-Wave travel faster and hence are the first waves to reach sesimograph. They propagate through gases, liguids or solids. \n",
# - S-Waves are attenuated by liquids.\n",
# - P-Waves are non-destructive, while both S-Waves and Surfcae Waves are destructive. \n",
# - Earthquake warning is possible if P-Waves are detected. Advanced warning time is dependent on the delay beween the arrival of P-wave and the arrival of the first destructive waves. This delay is a function of how deep the focus is, nature of earth layers and others. It ranges from few seconds to 90 seconds. Ground vibrations resulting from truck movement and contruction activitoes on earth shall be rejected for accurate detection of P-waves.
# 
# https://www.bgs.ac.uk/discoveringGeology/hazards/earthquakes/
# 
# ## Statistical Perspective
# 
# https://towardsdatascience.com/how-not-to-use-machine-learning-for-time-series-forecasting-avoiding-the-pitfalls-19f9d7adf424
# 
# We need to test data for random-walk/stochasiticity, e.g. correlation tests on both signal (f) and (df), covariance test?!
# 
# 

# # Recurrent Neural Network - LSTM
# 
# ## Part 1 - Data Preprocessing
# 
# ### Part 1.1 - [Filtering] + Downsampling

# In[13]:


# Importing the training set
"""Temporary: we downsample the training datatset to reduce time!"""

dataset_train = train_df.iloc[::params.down_sample,:]

print("Training will be performed on downsampled dataset which consists of ",dataset_train.shape[0],
      " examples out of the original number of training examples which is ", train_df.shape[0])

utility.plot_series(dataset_train, "Downsampled training series", params.results_dir)

training_set = dataset_train.values


# In[14]:


dataset_train.info()
dataset_train.head()


# In[15]:


# Import validation set
"""Temporary: we downsample the testing datatset to reduce time!"""
dataset_valid = valid_df.iloc[::params.down_sample, :]

print("Validation will be performed on truncated dataset which consists of ", dataset_valid.shape[0],
      " examples out of the original number of validation examples which is ", valid_df.shape[0])

final_valid_set = dataset_valid

utility.plot_series(final_valid_set, "Downsampled validation series", params.results_dir)

final_valid_set = final_valid_set.values


# In[16]:


dataset_valid.info()
dataset_valid.head()


# In[17]:


# Import test set
"""Temporary: we downsample the testing datatset to reduce time!"""
dataset_test = test_df.iloc[::params.down_sample, :]

print("Testing will be performed on truncated dataset which consists of ", dataset_test.shape[0],
      " examples out of the original number of test examples which is ", test_df.shape[0])

final_test_set = dataset_test

utility.plot_series(final_test_set, "Downsampled test series", params.results_dir)

final_test_set = final_test_set.values


# In[18]:


dataset_test.info()
dataset_test.head()


# ### Part 1.2 - Feature scaling

# In[19]:


# Feature Scaling
print('Scaling the datasets. scaling_type={}'.format(params.scaling_type))
t0 = time.time()
    
if params.scaling_type == 'None':
    training_set_scaled = training_set
    valid_set_scaled = final_valid_set
    test_set_scaled = final_test_set
else:
    if params.scaling_type == 'MinMaxScaler':
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif params.scaling_type == 'StandardScaler':
        scaler = StandardScaler()
    
    signal_scaled = scaler.fit_transform(training_set[:,0].reshape(-1,1))
    training_set_scaled = training_set.copy()   # May not be necessary
    training_set_scaled[:,0] = signal_scaled.reshape(-1)
    
    signal_scaled = scaler.fit_transform(final_valid_set[:,0].reshape(-1,1))
    valid_set_scaled = final_valid_set.copy()   # May not be necessary
    valid_set_scaled[:,0] = signal_scaled.reshape(-1)
    
    signal_scaled = scaler.fit_transform(final_test_set[:,0].reshape(-1,1))
    test_set_scaled = final_test_set.copy()   # May not be necessary
    test_set_scaled[:,0] = signal_scaled.reshape(-1)
    
    #valid_set_scaled = scaler.transform(final_valid_set)
    #test_set_scaled = scaler.transform(final_test_set)

print('Scaling complete. time_to_scale={:.2f} seconds'.format(time.time() - t0))


# ### Part 1.3 - Feature engineering (derivatives, log, etc)

# In[25]:


train_features = []
train_ys = []

valid_features = []
valid_ys = []

test_features = []
test_ys = []

for feature in params.features:
    train_feature = utility.get_feature(training_set_scaled, feature)
    train_features.append(train_feature.iloc[:,0])   # 0th column contains the feature
    train_ys.append(train_feature.iloc[:,1])   # 1st column contains y
    
    valid_feature = utility.get_feature(valid_set_scaled, feature)
    valid_features.append(valid_feature.iloc[:,0])
    valid_ys.append(valid_feature.iloc[:,1])
    
    test_feature = utility.get_feature(test_set_scaled, feature)
    test_features.append(test_feature.iloc[:,0])
    test_ys.append(test_feature.iloc[:,1])

utility.trim_to_same_length(train_features, train_ys)
utility.trim_to_same_length(valid_features, valid_ys)
utility.trim_to_same_length(test_features, test_ys)

train_ys = train_ys[0]   # Because all the ys dataframes in this list are the same
valid_ys = valid_ys[0]
test_ys = test_ys[0]

# for i, feature in enumerate(params.features):
#     print(feature)
#     print("X_shape = {}".format(train_features[i].shape))
#     print("y_shape = {}".format(train_ys[i].shape))


# ### Part 1.4 - Preparing time-windowed data matrix (to feed to the LSTM)

# In[37]:


# Creating the training dataset (X_train and y_train)
# X_train is a numpy array with some no. of examples. Each example is a seismic signal window of length time_steps
# y_train has the same no. of examples. Each example is the time_to_eq value that corresponds to the last element of seismic signal window (just 1 value)

# ToDo:
# Draw a diagram here illustrating how this input is prepared.
# Write an equation for no. of examples as a function of (training_signal_length, time_steps, stride)

print('Preparing input to the RNN (training set)')
t0 = time.time()

X_train = []
y_train = []

num_features = len(params.features)

for i in range (params.time_steps, train_ys.shape[0], params.window_stride):
    X_window = np.zeros((params.time_steps, num_features))
    for j, feature_df in enumerate(train_features):
        X_window[:, j] = feature_df.iloc[i - params.time_steps:i]
    
    X_train.append(X_window)
    y_train.append(train_ys.iloc[i])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping since RNN accepts 3d input
X_train = np.reshape (X_train, (X_train.shape[0], X_train.shape[1], num_features))
y_train = np.reshape (y_train, (-1, 1))   # Not required when we are predicting just one output value, but necessary when we predict more
print ("The 3d shape necessary for RNN's input is ", X_train.shape, " . Note how the number of examples is reduced by the defined time steps, i.e. ", params.time_steps)

assert X_train.shape[1] == params.time_steps

print('Preparing input complete. time_to_prepare={:.2f} seconds'.format(time.time() - t0))

print(X_train.shape)
print(y_train.shape)


# In[39]:


X_valid = []
y_valid = []

for i in range(params.time_steps, valid_ys.shape[0], params.window_stride):
    X_window = np.zeros((params.time_steps, num_features))
    for j, feature_df in enumerate(valid_features):
        X_window[:, j] = feature_df.iloc[i - params.time_steps:i]
    
    X_valid.append(X_window)
    y_valid.append(valid_ys.iloc[i])

X_valid = np.array(X_valid)
X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], num_features))
y_valid = np.reshape (y_valid, (-1, 1))   # Not required when we are predicting just one output value, but necessary when we predict more

print(X_valid.shape)
print(y_valid.shape)


# In[41]:


X_test = []
y_test = []

for i in range(params.time_steps, test_ys.shape[0], params.window_stride):
    X_window = np.zeros((params.time_steps, num_features))
    for j, feature_df in enumerate(test_features):
        X_window[:, j] = feature_df.iloc[i - params.time_steps:i]
    
    X_test.append(X_window)
    y_test.append(test_ys.iloc[i])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_features))
y_test = np.reshape(y_test, (-1, 1))   # Not required when we are predicting just one output value, but necessary when we predict more

print(X_test.shape)
print(y_test.shape)


# ## Part 2 - Building the RNN

# In[42]:


# Initialising the RNN
regressor = Sequential ()

# Adding the hidden layers as given in the parameters

for i, (units, dropout_rate) in enumerate(zip(params.rnn_layer_units, params.rnn_layer_dropout_rate)):
    # Common args for all layers
    input_shape = (None,)
    return_sequences = True
    
    # Set special args for first and last layer
    if i == 0:  # First hidden layer
        input_shape = (params.time_steps, num_features)
    if i == len(params.rnn_layer_units) - 1:   # Last hidden layer
        return_sequences = False
        
    regressor.add(LSTM(units=units, return_sequences=return_sequences, input_shape=input_shape))
    regressor.add (Dropout(dropout_rate))

# Adding the output layer
regressor.add (Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.summary()


# ## Part 3 - Training the RNN

# In[44]:


print('Training the RNN with the training set')
t0 = time.time()

#with tf.device('/cpu:0'):

earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')

history = regressor.fit(X_train, y_train, epochs=params.epochs, batch_size=params.batch_size, validation_data=(X_valid, y_valid), callbacks=[earlyStopping], verbose=2)

time_to_train = time.time() - t0
print('Training complete. time_to_train={:.2f} seconds ({:.2f} minutes)'.format(time_to_train, time_to_train/60))


# In[45]:


# Save the final trained model (in case we need to continue training from this point on)

model_filepath = params.results_dir + '/final_model.h5'
regressor.save(model_filepath, overwrite=True)

print('RNN model saved to {}'.format(model_filepath))


# In[46]:


utility.plot_training_history(history, params.results_dir)

model_plot_filename = params.results_dir + '/' + 'rnn_plot.png'
plot_model(regressor, to_file=model_plot_filename, show_shapes=True, show_layer_names=True)

print('RNN plot saved to {}'.format(model_plot_filename))


# ## Part 4 - Making the predictions and visualising the results
# 
# ### Part 4.1 - Predicting on the training set

# In[47]:


# Predict on training set

print('Predicting on the training set using the trained RNN')
t0 = time.time()
train_predicted_time = regressor.predict(X_train)

train_predicted_time_orig = train_predicted_time.copy()  # To plot

train_predicted_time, y_train = utility.ma_filter(train_predicted_time.reshape((-1,)), y_train, params.ma_window)

#predicted_time = sc.inverse_transform(predicted_time)
print('Predicting on the training set complete. time_to_predict={:.2f} seconds'.format(time.time() - t0))


# In[48]:


# Save predictions on training set

train_prediction = pd.DataFrame(train_predicted_time)
train_pred_filename = params.results_dir + '/' + 'train_prediction.csv'
train_prediction.to_csv(train_pred_filename)

train_prediction_orig = pd.DataFrame(train_predicted_time_orig)
train_pred_orig_filename = params.results_dir + '/' + 'train_prediction_orig.csv'
train_prediction_orig.to_csv(train_pred_orig_filename)

print('Predictions on train set saved to {}, and {}'.format(train_pred_filename, train_pred_orig_filename))


# In[49]:


# Visualize predictions on training set

train_res_orig_plot_filename = params.results_dir + '/' + 'train_true_vs_pred_orig' + '.png'
utility.plot_results(y_train, train_prediction_orig, 'True (orig) vs predicted time_to_earthquake on train set', train_res_orig_plot_filename)

train_res_plot_filename = params.results_dir + '/' + 'train_true_vs_pred' + '.png'
utility.plot_results(y_train, train_prediction, 'True vs predicted time_to_earthquake on train set', train_res_plot_filename)


# In[50]:


# Compute error metrics on training set

N = params.ma_window
train_mse, train_rmse, train_mae, train_r2 = utility.metrics(y_train, train_predicted_time_orig[N-1:])

print('Error metrics on training set without MA filter (original). train_mse: {:.4f}, train_rmse: {:.4f}, train_mae: {:.4f}, train_r2: {:.4f}'.
      format(train_mse, train_rmse, train_mae, train_r2))

train_mse, train_rmse, train_mae, train_r2 = utility.metrics(y_train, train_predicted_time)

print('Error metrics on training set with MA filter (filtered). train_mse: {:.4f}, train_rmse: {:.4f}, train_mae: {:.4f}, train_r2: {:.4f}'.
      format(train_mse, train_rmse, train_mae, train_r2))


# ### Part 4.2 - Predicting on the validation set

# In[51]:


# Predict on validation set

print('Predicting on the validation set using the trained RNN')
t0 = time.time()
valid_predicted_time = regressor.predict(X_valid)

valid_predicted_time_orig = valid_predicted_time.copy()

valid_predicted_time, y_valid = utility.ma_filter(valid_predicted_time.reshape((-1,)), y_valid, params.ma_window)

print('Predicting on the validation set complete. time_to_predict={:.2f} seconds'.format(time.time() - t0))


# In[52]:


# Save predictions on validation set

valid_prediction = pd.DataFrame(valid_predicted_time)
valid_pred_filename = params.results_dir + '/' + 'validation_prediction.csv'
valid_prediction.to_csv(valid_pred_filename)

valid_prediction_orig = pd.DataFrame(valid_predicted_time_orig)
valid_pred_orig_filename = params.results_dir + '/' + 'validation_prediction_orig.csv'
valid_prediction_orig.to_csv(valid_pred_orig_filename)

print('Predictions on validation set saved to {}, and {}'.format(valid_pred_filename, valid_pred_orig_filename))


# In[53]:


# Visualize predictions on validation set

valid_res_orig_plot_filename = params.results_dir + '/' + 'validation_true_vs_pred_orig' + '.png'
utility.plot_results(y_valid, valid_prediction_orig, 'True (orig) vs predicted time_to_earthquake on validation set', valid_res_orig_plot_filename)

valid_res_plot_filename = params.results_dir + '/' + 'validation_true_vs_pred' + '.png'
utility.plot_results(y_valid, valid_prediction, 'True vs predicted time_to_earthquake on validation set', valid_res_plot_filename)


# In[55]:


# Compute error metrics on validation set

valid_mse, valid_rmse, valid_mae, valid_r2 = utility.metrics(y_valid, valid_predicted_time_orig[N-1:])

print('Error metrics on validation set without MA filter (original). valid_mse: {:.4f}, valid_rmse: {:.4f}, valid_mae: {:.4f}, valid_r2: {:.4f}'.
      format(valid_mse, valid_rmse, valid_mae, valid_r2))

valid_mse, valid_rmse, valid_mae, valid_r2 = utility.metrics(y_valid, valid_predicted_time)

print('Error metrics on validation set with MA filter (filtered). valid_mse: {:.4f}, valid_rmse: {:.4f}, valid_mae: {:.4f}, valid_r2: {:.4f}'.
      format(valid_mse, valid_rmse, valid_mae, valid_r2))


# ### Part 4.3 - Predicting on the test set

# In[56]:


# Predict on test set

print('Predicting on the test set using the trained RNN')
t0 = time.time()
test_predicted_time = regressor.predict(X_test)

test_predicted_time_orig = test_predicted_time.copy()

test_predicted_time, y_test = utility.ma_filter(test_predicted_time.reshape((-1,)), y_test, params.ma_window)

print('Predicting on the test set complete. time_to_predict={:.2f} seconds'.format(time.time() - t0))


# In[57]:


# Save predictions on test set

test_prediction = pd.DataFrame(test_predicted_time)
test_pred_filename = params.results_dir + '/' + 'test_prediction.csv'
test_prediction.to_csv(test_pred_filename)

test_prediction_orig = pd.DataFrame(test_predicted_time_orig)
test_pred_orig_filename = params.results_dir + '/' + 'test_prediction_orig.csv'
test_prediction_orig.to_csv(test_pred_orig_filename)

print('Predictions on validation set saved to {}, and {}'.format(test_pred_filename, test_pred_orig_filename))


# In[58]:


# Visualize predictions on test set

test_res_orig_plot_filename = params.results_dir + '/' + 'test_true_vs_pred_orig' + '.png'
utility.plot_results(y_test, test_prediction_orig, 'True (orig) vs predicted time_to_earthquake on test set', test_res_orig_plot_filename)

test_res_plot_filename = params.results_dir + '/' + 'test_true_vs_pred' + '.png'
utility.plot_results(y_test, test_prediction, 'True vs predicted time_to_earthquake on test set', test_res_plot_filename)


# In[59]:


# Compute error metrics on test set

test_mse, test_rmse, test_mae, test_r2 = utility.metrics(y_test, test_predicted_time_orig[N-1:])

print('Error metrics on test set without MA filter (original). test_mse: {:.4f}, test_rmse: {:.4f}, test_mae: {:.4f}, test_r2: {:.4f}'.
      format(test_mse, test_rmse, test_mae, test_r2))

test_mse, test_rmse, test_mae, test_r2 = utility.metrics(y_test, test_predicted_time)

print('Error metrics on test set with MA filter (filtered). test_mse: {:.4f}, test_rmse: {:.4f}, test_mae: {:.4f}, test_r2: {:.4f}'.
      format(test_mse, test_rmse, test_mae, test_r2))


# In[ ]:


# Save the output (results) of this notebook to the results_dir folder
# get_ipython().run_line_magic('sx', 'jupyter nbconvert --to html --output-dir=$params.results_dir --TemplateExporter.exclude_input=True LANL_NB.ipynb')


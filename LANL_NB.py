
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# To visualize the RNN network
from keras.utils.vis_utils import plot_model
import pydot

import utility  # Contains various helper utility functions


# In[2]:


# Hardware (GPU or CPU)

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # Disable GPU as it appears to be slower than CPU (to enable GPU, comment out this line and restart the kernel)

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

# LSTM network architecture
params.time_steps = 50
params.rnn_layer_units = [10, 5, 2]   # The length of this list = no. of hidden layers
params.rnn_layer_dropout_rate = [0.2, 0.2, 0]   # Dropout rate for each layer (0 for no dropout)

# Training
params.epochs = 2
params.batch_size = 64

# Admin variables unrelated to RNN parameters
params.results_dir = 'results/current_run'   # Folder to save results
params.experiment_num = -1   # For bookkeeping; currently not used


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


# In[9]:


utility.print_info(train_df)

if do_plot_series:
    utility.plot_series(train_df, "Training series", params.results_dir) # This is time and memory consuming. Do not run this cell unless necessary


# In[10]:


utility.print_info(valid_df)

if do_plot_series:
    utility.plot_series(valid_df, "Validation series", params.results_dir) # This is time and memory consuming. Do not run this cell unless necessary


# In[11]:


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

# In[12]:


#train_df.max()


# In[13]:


#train_df.min()


# # Recurrent Neural Network - LSTM
# 
# ## Part 1 - Data Preprocessing

# In[10]:


# Importing the training set
"""Temporary: we downsample the training datatset to reduce time!"""

dataset_train = train_df.iloc[::params.down_sample,:]
training_set = dataset_train.iloc[:, 0:2].values
print("Training will be performed on downsampled dataset which consists of ",dataset_train.shape[0],
      " examples out of the original number of training examples which is ", train_df.shape[0])

utility.plot_series(dataset_train, "Downsampled training series", params.results_dir)


# In[15]:


dataset_train.info()
dataset_train.head()


# In[13]:


# Feature Scaling
print('Scaling the training set. scaling_type={}'.format(params.scaling_type))
t0 = time.time()
    
if params.scaling_type == 'None':
    training_set_scaled = training_set
else:
    if params.scaling_type == 'MinMaxScaler':
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif params.scaling_type == 'StandardScaler':
        scaler = StandardScaler()
    
    signal_scaled = scaler.fit_transform(training_set[:,0].reshape(-1,1))

    training_set_scaled = training_set.copy()   # May not be necessary
    training_set_scaled[:,0] = signal_scaled.reshape(-1)

print('Scaling complete. time_to_scale={:.2f} seconds'.format(time.time() - t0))


# In[14]:


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
    
for i in range (params.time_steps, training_set_scaled.shape[0]): 
    X_train.append (training_set_scaled[i - params.time_steps:i, 0])
    y_train.append (training_set_scaled[i, 1])
X_train, y_train = np.array (X_train), np.array (y_train)

# Reshaping since RNN accepts 3d input
X_train = np.reshape (X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape (y_train, (-1, 1))   # Not required when we are predicting just one output value, but necessary when we predict more
print ("The 3d shape necessary for RNN's input is ", X_train.shape, " . Note how the number of examples is reduced by the defined time steps, i.e. ", params.time_steps)

assert X_train.shape[1] == params.time_steps

print('Preparing input complete. time_to_prepare={:.2f} seconds'.format(time.time() - t0))


# In[15]:


#Check
#print(training_set_scaled[99,0], " ", training_set_scaled[100,1])  # Gives errors when time_steps != 100 (fix this bug)
#print(X_train[0,99,0]," ", y_train[0])


# ## Part 2 - Building the RNN

# In[16]:


# Initialising the RNN
regressor = Sequential ()

# Adding the hidden layers as given in the parameters

for i, (units, dropout_rate) in enumerate(zip(params.rnn_layer_units, params.rnn_layer_dropout_rate)):
    # Common args for all layers
    input_shape = (None,)
    return_sequences = True
    
    # Set special args for first and last layer
    if i == 0:  # First hidden layer
        input_shape = (params.time_steps, 1)
    if i == len(params.rnn_layer_units) - 1:   # Last hidden layer
        return_sequences = False
        
    regressor.add(LSTM(units=units, return_sequences=return_sequences, input_shape=input_shape))
    regressor.add (Dropout(dropout_rate))

# Adding the output layer
regressor.add (Dense (units=1))

# Compiling the RNN
regressor.compile (optimizer='adam', loss='mean_squared_error')
regressor.summary()


# ## Part 3 - Training the RNN

# In[17]:


print('Training the RNN with the training set')
t0 = time.time()

#with tf.device('/cpu:0'):

history = regressor.fit (X_train, y_train, epochs=params.epochs, batch_size=params.batch_size)

time_to_train = time.time() - t0
print('Training complete. time_to_train={:.2f} seconds ({:.2f} minutes)'.format(time_to_train, time_to_train/60))


# In[18]:


# Save the final trained model (in case we need to continue training from this point on)

model_filepath = params.results_dir + '/final_model.h5'
regressor.save(model_filepath, overwrite=True)

print('RNN model saved to {}'.format(model_filepath))


# In[20]:


utility.plot_training_history(history, params.results_dir)

model_plot_filename = params.results_dir + '/' + 'rnn_plot.png'
plot_model(regressor, to_file=model_plot_filename, show_shapes=True, show_layer_names=True)

print('RNN plot saved to {}'.format(model_plot_filename))


# ## Part 4 - Making the predictions and visualising the results

# In[21]:


# Import validation set
"""Temporary: we downsample the testing datatset to reduce time!"""
dataset_test = valid_df.iloc[::params.down_sample,:]
true_test_time = dataset_test.iloc[:,1].values
print("Validation will be performed on truncated dataset which consists of ", dataset_test.shape[0],
      " examples out of the original number of training examples which is ", valid_df.shape[0])


# In[38]:


dataset_test.info()
dataset_test.head()


# In[23]:


#Because we have time_steps time steps and we we want to predict the first entry of time_to_failure in the validation set, we have to look back time_steps samples. 
#Hence, we get these time_steps past samples from the training set. This is why we first concatenate both training and validation. This step may be omitted if we just need to predict one value
#for the whole test set (such as in the provided test files where one value is only needed so we can look back in the same data provided ) 
dataset_total = pd.concat((dataset_train['acoustic_data'], dataset_test['acoustic_data']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - params.time_steps:].values
inputs = inputs.reshape(-1,1)

# Feature Scaling
if params.scaling_type == 'None':
    inputs_scaled = inputs
else:
    print('Scaling the inputs set. scaling_type={}'.format(params.scaling_type))
    t0 = time.time()
    inputs_scaled = scaler.transform(inputs) 
    print('Scaling complete. time_to_scale={:.2f} seconds'.format(time.time() - t0))

inputs_scaled.shape # So we end up with input size = size of validation set + time_steps


# In[24]:


X_test = []

for i in range(params.time_steps, inputs_scaled.shape[0]):
    X_test.append(inputs_scaled[i-params.time_steps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test.shape


# In[25]:


# Predict on training set

print('Predicting on the training set using the trained RNN')
t0 = time.time()
train_predicted_time = regressor.predict(X_train)
#predicted_time = sc.inverse_transform(predicted_time)
print('Predicting on the training set complete. time_to_predict={:.2f} seconds'.format(time.time() - t0))


# In[26]:


# Save predictions on training set

train_prediction = pd.DataFrame(train_predicted_time)
train_pred_filename = params.results_dir + '/' + 'train_prediction.csv'
train_prediction.to_csv(train_pred_filename)
print('Predictions on train set saved to {}'.format(train_pred_filename))


# In[27]:


# Visualize predictions on training set

true_train_time = dataset_train.iloc[:,1].values
true_train_time = true_train_time[params.time_steps:]   # Ignore the first time_steps values (because predictions are not available for those)

train_res_plot_filename = params.results_dir + '/' + 'train_true_vs_pred' + '.png'
utility.plot_results(true_train_time, train_prediction, 'True vs predicted time_to_earthquake on train set', train_res_plot_filename)


# In[28]:


# Compute error metrics on training set

train_mse = mean_squared_error(true_train_time, train_predicted_time)
train_rmse = train_mse ** 0.5
train_mae = mean_absolute_error(true_train_time, train_predicted_time)

print('Error metrics on test set. train_mse: {:.4f}, train_rmse: {:.4f}, train_mae: {:.4f}'.format(train_mse, train_rmse, train_mae))


# In[29]:


# Predict on test set

print('Predicting on the test set using the trained RNN')
t0 = time.time()
test_predicted_time = regressor.predict(X_test)
#predicted_time = sc.inverse_transform(predicted_time)
print('Predicting on the test set complete. time_to_predict={:.2f} seconds'.format(time.time() - t0))


# In[30]:


# Save predictions on test set

test_prediction = pd.DataFrame(test_predicted_time)
test_pred_filename = params.results_dir + '/' + 'test_prediction.csv'
test_prediction.to_csv(test_pred_filename)
print('Predictions on test set saved to {}'.format(test_pred_filename))


# In[33]:


# Visualize predictions on test set

test_res_plot_filename = params.results_dir + '/' + 'test_true_vs_pred' + '.png'
utility.plot_results(true_test_time, test_prediction, 'True vs predicted time_to_earthquake on test set', test_res_plot_filename)


# In[34]:


# Compute error metrics on test set

test_mse = mean_squared_error(true_test_time, test_predicted_time)
test_rmse = test_mse ** 0.5
test_mae = mean_absolute_error(true_test_time, test_predicted_time)

print('Error metrics on test set. test_mse: {:.4f}, test_rmse: {:.4f}, test_mae: {:.4f}'.format(test_mse, test_rmse, test_mae))


# In[35]:


# Save the output (results) of this notebook to the results_dir folder
get_ipython().run_line_magic('sx', 'jupyter nbconvert --to html --output-dir=$params.results_dir --TemplateExporter.exclude_input=True LANL_NB.ipynb')


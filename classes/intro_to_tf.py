import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

'''
1. Using LinearRegressor class in TF to predict median
    housing price at a granularity of city blocks, based on one input feature
2. Eval the accuracy of model's prediction by using Root Mean Squared Error (RMSE)
3. Improve accuracy of model by tuning its hyperparameters
'''

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# Load the data set
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

'''
We'll randomize the data now, just so that there's no ordering that effects the performance of
SGD (Stochastic Gradient Descent), and let's use median_house_value in units of 1k, easier with learning rates in a range
tha we use normally
'''

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe['median_house_value'] /= 1000.0

# This will tell you the column descriptors and give you some mean, stdev, max, min, etc.
# california_housing_dataframe.describe()

'''
TF Model
We are going to be using total_rooms as our input feature
TO train our model, we're going to be using LinearRegressor interface provided by TF Estimator API
'''

'''
Step 1) Define Features & Configure Feature Columns
We need to specify the type of data each feature contains for TensorFlow
    1. Categorical Data - Data that is textual, we're not going to use this for this example...but it could be style of hoome
    2. Numerical Data - Data that is a number (int or float) and you want to treat as a number.

To indicate feature's data type, use a construct called a feature column. They store only a description of the feature data
 and not the feature data itself.
'''
my_feature = california_housing_dataframe[["total_rooms"]]
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

'''
Step 2) Define the Target
Target is going to be median_house_value
'''

targets = california_housing_dataframe["median_house_value"]

'''
Step 3) Configure the LinearRegressor
We'll configure a linear regression model using LinearRegressor -- we'll train this model using the
 GradientDescentOptimizer, which implements Mini-Batch SGD.
 learning_rate argument controls the size of the gradient step

Gradient clipping - ensures the magnitude of the gradient doesn't get too large during the training. 
We're going to be using it via "clip_gradients_by_norm"
'''
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
# 5.0 is the clip_norm, which is basically a scalar Tensor
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# COnfigure LinearRegression Model with our feature columns & optimizer
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer,
)

'''
Step 4) Define the Input Function
To import the housing data into our LinearRegressor, we'll need to define an input func that instructs TF how 
to pre-process the data, batch, shuffle and repeat during model training

Convert our pandas feature data into dict of NumPy arrays, then use TF Dataset API to construct dataset obj
from our data, break into batches of batch_size, to be repeated for a specified number of epochs (num_epochs)
default value for num_epochs is None, which means the input data will be repeated indefinitely
'''

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    '''
    Trains the linear regression model of one feature

    :param features: pandas DataFrame of features
    :param targets: pandas DataFrame of targets
    :param batch_size: size of batches to be passed to the model
    :param shuffle: whether or not to shuffle the data
    :param num_epochs: number of epochs for which the data should be repeated
    :return:
        Tuple of (features, labels) for next data batch
    '''

    # Convert pandas data into dict of np arrays
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Dataset and configure batching / repeating
    ds = Dataset.from_tensor_slices((features, targets))  # 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        # buffer_size specifies the size of the data that will be randomly shuffled
        ds = ds.shuffle(buffer_size=10000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

'''
Step 5) Train the Model
We'll call train() on our linear_regressor to train the model, wrap my_input_fn in a lambda so we can pass in 
my_feature and target as arguments...to start we'll train for 100 steps
'''
# TODO: read https://www.tensorflow.org/get_started/datasets_quickstart#passing_input_fn_data_to_your_model
_ = linear_regressor.train(
    input_fn=lambda: my_input_fn(my_feature, targets),
    steps=100
)

'''
Step 6) Evaluate the Model
Let's make predictions on the training data and see how well our model fit it during training.
'''

# Create input fn for predictions
# Since we're making one prediction per example, don't need to repeat / shuffle
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# call predict() on linear_regressor
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# format predictions as NumPy array so we can calculator error metrics
predictions = np.array([item['predictions'][0] for item in predictions])

# Print MSE and RMSE
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)

print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
# Mean Squared Error (on training data): 56367.025
# Root Mean Squared Error (on training data): 237.417

min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)
'''
Mean Squared Error (on training data): 56367.025
Root Mean Squared Error (on training data): 237.417
Min. Median House Value: 14.999
Max. Median House Value: 500.001
Difference between Min. and Max.: 485.002
Root Mean Squared Error: 237.417

Our RMSE spans nearly half of the housing prices, are we able to do better?
Let's look at how well our predictions match our targets in summary metrics
'''

calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
print(calibration_data.describe())
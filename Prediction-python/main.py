import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib as plt

tf.disable_v2_behavior()


GOLD_TRAIN_DATA = 'CSV_Files/Gold Futures Year.csv'
GOLD_TEST_DATA = 'CSV_Files/Gold Futures Month.csv'

current_train_data = GOLD_TRAIN_DATA
current_test_data = GOLD_TEST_DATA

# Number of lines to retrieve from csv files
NUM_TRAIN_DATA_POINTS = 331
NUM_TEST_DATA_POINTS = 23

# Rate our model is going to change the values found at W and b, to optimize or minimize our loss
LEARNING_RATE = 0.1


# Load Data we use from our csv files and return final and opening prices, also volume for each day.
def load_stock_data(stock_name, num_data_points):
    data = pd.read_csv(stock_name,
                       skiprows=0,
                       nrows=num_data_points,
                       usecols=['Price', 'Open', 'Vol.'])
    # Price of stock at the end of each day
    final_prices = data['Price'].astype(str).str.replace(',', '').astype(np.float)
    # Price of stock at the beginning of the day
    opening_prices = data['Open'].astype(str).str.replace(',', '').astype(np.float)
    # Volume of stock exchange throughout the day
    volumes = data['Vol.'].str.strip('MK').astype(np.float)
    return final_prices, opening_prices, volumes


def calculate_price_differences(final_prices, opening_prices): #iterate through list and take the differences
    price_differences = []
    # array to store the float price differences

    for d_i in range(len(final_prices) - 1):

        price_difference = opening_prices[d_i + 1] - final_prices[d_i]
        # take diff between opening price of next day and final price of current day
        price_differences.append(price_difference)

    return price_differences


# finals, openings, volumes = load_stock_data(current_test_data, NUM_TEST_DATA_POINTS)
# print(calculate_price_differences(finals, openings))


# y = Wx +b
# Training and testing role model
x = tf.placeholder(tf.float32, name = 'x')
# Variable we want our model to optimize or to train - model will optimize it
W = tf.Variable([.1], name = 'W')
b = tf.Variable([.1], name='b')
# Global variable initializer
y = W * x + b

# What we except our model will be aiming, it's the actual value
y_predicted = tf.placeholer(tf.float32, name='y_predicted')

# Sum of all the differences between the expected values (y_predicted) and the actual values (y).
# Square them to reduction the negative values
loss = tf.reduce_sum(tf.square(y - y_predicted))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
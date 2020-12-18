import tensorflow as tf
from matplotlib import ticker
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque

import os
import numpy as np
import pandas as pd
import random

# Basic setup seed, geet same results after running multiple times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


# Download Dataset and processing

def shuffle_in_unison(a, b):
    # shuffle 2 arrays
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.shuffle(b)


"""
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params from Documentation:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        split_by_date (bool): whether we split the dataset into training/testing by date, setting it 
            to False will split datasets in a random way
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """


def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True, test_size=0.2,
              features_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    # Check if market is loaded from yahoo
    if isinstance(ticker, str):
        # then load it from yahoo_fin api
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # if downloaded then use it
        df = ticker
    else:
        raise TypeError("ticker can be a str or a pd.DataFrame instances only.")
    # Elements we want to return from function above
    result = {}
    # and df
    result['df'] = df.copy()
    # Verification of passed_features
    for col in feature_columns:
        assert col in df.columns, f"{col} doest not exist in the dataframe"
    # show date in column
    if "date" not in df.columns:
        df["date"] = df.index
    if scale:
        column_scaler = {}
        # Scale data between 0 and 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # Add Min Max Scaler for the result
        result["column_scaler"] = column_scaler
    df['future'] = df['adjclose'].shift(-lookup_step)
    # Last step will give NaN for future columns, so we get them before the NaN drop
    last_sequence = np.array(df[feature_colums].tail(lookup_step))
    df.dropna(inplace=True)
    sequence_data = []
    sequence = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["date"]].values, df[future].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the datase

    # To do :
    # Add last_sequence list and array
    # Add formula X, Y,
    # Train and Test
    # ...
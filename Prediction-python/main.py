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
    for col in features_columns:
        assert col in df.columns, f"{col} doest not exist in the dataframe"
    # show date in column
    if "date" not in df.columns:
        df["date"] = df.index
    if scale:
        column_scaler = {}
        # Scale data between 0 and 1
        for column in features_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # Add Min Max Scaler for the result
        result["column_scaler"] = column_scaler
    df['future'] = df['adjclose'].shift(-lookup_step)
    # Last step will give NaN for future columns, so we get them before the NaN drop
    last_sequence = np.array(df[features_columns].tail(lookup_step))
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[features_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the datase
    last_sequence = list([s[:len(features_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result['last_sequence'] = last_sequence

    ## Add X, Y
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    if split_by_date:
        # Training and Testing set
        train_samples = int((1-test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        if shuffle:
            #Shuffle for training
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
        else:
            result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

        #get the list of test set dates
        dates = result["X_train"][:, -1, -1]
        result["test_df"] = result["df"].loc[dates]
        #Remove the columns dates, convert to float32
        result["X_train"] = result["X_train"][:, :, :len(features_columns)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :, :len(features_columns)].astype(np.float32)
        return result


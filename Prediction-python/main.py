import tensorflow as tf
from matplotlib import ticker
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
import time
import os
import numpy as np
import pandas as pd
import random

# Basic setup seed, get same results after running multiple times
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
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        if shuffle:
            # Shuffle for training
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
        else:
            result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                        test_size=test_size,
                                                                                                        shuffle=shuffle)

        # get the list of test set dates
        dates = result["X_train"][:, -1, -1]
        result["test_df"] = result["df"].loc[dates]
        # Remove the columns dates, convert to float32
        result["X_train"] = result["X_train"][:, :, :len(features_columns)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :, :len(features_columns)].astype(np.float32)
        return result


# Creating model

def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                 loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True),
                                        batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

# Window size or the sequence length

N_STEPS = 50
#Look up step, 1 is the next day
LOOKUP_STEP = 15
# whether to scale feature columns & output price
SCALE = True
scale_str = f"sc-{int(SCALE)}"
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"
#wether to split the training/testing set by date
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
#test ratio size
TEST_SIZE  = 0.2
# What columns we use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# Date
date_now = time.strftime("%Y-%m-%d")

# Model parameters
N_LAYERS = 2
#LSTM cell
CELL = LSTM
# Neurones, fitting best combination
UNITS = 256
# rate of possible not training node in layer
DROPOUT = 0.4
# Bidi
BIDIRECTIONAL = False
#training parameters
LOSS = "mae"
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZZ = 64
EPOCH = 500

# Trying with TSLA stock market

ticker = "TSLA"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"











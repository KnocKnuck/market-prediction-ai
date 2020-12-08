import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as plt

GOLD_TRAIN_DATA = 'CSV_Files/Gold Futures Year.csv'
GOLD_TEST_DATA = 'CSV_Files/Gold Futures Month.csv'

current_train_data = GOLD_TRAIN_DATA
current_test_data = GOLD_TEST_DATA

# Number of lines to retrieve from csv files
NUM_TRAIN_DATA_POINTS = 331
NUM_TEST_DATA_POINTS = 23


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
    # Volume of stock exchange throught the day
    volumes = data['Vol.'].str.strip('MK').astype(np.float)
    return final_prices, opening_prices, volumes


def calculate_price_differences(final_prices, opening_prices):
    price_diffs = []
    for d_in in range(len(final_prices) - 1):
        price_diff = opening_prices[d_in + 1] - final_prices[d_in]
        price_diffs.append(price_diff)
        return price_diffs


finals, openings, volumes = load_stock_data(current_test_data, NUM_TEST_DATA_POINTS)
print(calculate_price_differences(finals, openings))

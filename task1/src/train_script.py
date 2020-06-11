import os
import sys
import time
import pandas as pd
from task1.src.Classification import *
from task1.src.Regression import *
from task1.src.preprocess_fllght_data import *
from task1.src.weather_preprocess import *

NUM_OF_ARGS = 3

PATH_TO_TRAIN_DATA = "../data/train_data.csv"
PATH_TO_1K_TRAIN_DATA = "../data/data1k.csv"
PATH_TO_TEST_DATA = "../data/train_data.csv"
PATH_TO_WEATHER_DATA = "../data/all_weather_data.csv"

MY_DIR = os.path.dirname(__file__)
PATH_TO_WEATHER_FILE_PATH = os.path.join(MY_DIR, '..', 'data', 'all_weather_data.csv')
PATH_TO_WEATHER_JFK_PATH = os.path.join(MY_DIR, '..', 'small_data', 'weather_jfk.csv')
PATH_TO_TRAIN_DATA_FILE_PATH = os.path.join(MY_DIR, '..', 'data', 'train_data.csv')
PATH_TO_SMALL_TRAIN_DATA_PATH = os.path.join(MY_DIR, '..', 'small_data', 'data1k.csv')
PATH_TO_MEDIUM_TRAIN_DATA_PATH = os.path.join(MY_DIR, '..', 'small_data', 'data10k.csv')

WEATHER_FILE = {
    'all_weather': PATH_TO_WEATHER_FILE_PATH,
    'jfk': PATH_TO_WEATHER_JFK_PATH
}

TRAIN_DATA_FILE = {
    '1k': PATH_TO_SMALL_TRAIN_DATA_PATH,
    '10k': PATH_TO_MEDIUM_TRAIN_DATA_PATH,
    'all_data': PATH_TO_TRAIN_DATA_FILE_PATH
}


def is_valid_usage():
    return len(sys.argv) == NUM_OF_ARGS and sys.argv[1] in TRAIN_DATA_FILE \
           and sys.argv[2] in WEATHER_FILE


def run_classifier(X_test: pd.DataFrame, X_train: pd.DataFrame, y_test_type: pd.DataFrame,
                   y_train_type: pd.DataFrame):
    """
    Run the classifier, and print the score of the trained model
    :param X_test: Test feature matrix
    :param X_train: Train feature matrix
    :param y_test_type: Test types
    :param y_train_type: Train types
    """
    start = time.time()
    # Get reg model
    mask_test = y_test_type.notnull()
    mask_train = y_train_type.notnull()

    y_test_type = y_test_type.astype('category')
    y_train_type = y_train_type.astype('category')
    y_test_type, y_train_type = y_test_type.cat.codes, y_train_type.cat.codes
    y_test_type, y_train_type = y_test_type + 1, y_train_type + 1

    class_model = get_classification_model(X_train[mask_train], y_train_type[mask_train],
                                           X_test[mask_test],
                                           y_test_type[mask_test])
    print(class_model.to_string())
    end = time.time()
    print("run classifier time: {}".format(end - start))


def run_regression(X_test, X_train, y_test_delay, y_train_delay):
    """
    Run the regression, and print the score of the trained model
    :param X_test: Test feature matrix
    :param X_train: Train feature matrix
    :param y_test_delay: Test delay
    :param y_train_delay: Train delay
    """
    start = time.time()

    print("run reg models")
    reg_model = get_reg_model(X_train, y_train_delay, X_test, y_test_delay)
    print(reg_model.to_string())

    end = time.time()
    print("run reg time: {}".format(end - start))


def get_feature_matrix(train_path: str, weather_path: str):
    """
    Get the preprocessed data
    :param train_path: The train data file path
    :param weather_path: The weather data file path
    :return:  design_matrix, y_delay, y_type
    """
    start = time.time()

    if os.path.isfile("../pickle/X_10000.csv"):
        design_matrix, y_delay, y_type = read_saved_files()

    else:
        print('load data')
        df = pd.read_csv(train_path,
                         dtype={'FlightDate': str, 'CRSDepTime': str, 'CRSArrTime': str},
                         nrows=100000)
        print('load weather')
        weather_df = pd.read_csv(weather_path, low_memory=False)
        print('preprocess weather')
        df = preprocess_weather_data(df, weather_df)
        print('preprocess data')
        design_matrix, y_delay, y_type = preprocess_flight_data(df)
        end = time.time()
        print("load data time: {}".format(end - start))

        design_matrix.info()
        print(design_matrix.describe().to_string())
        print(design_matrix.head(50).to_string())

        save_to_file(design_matrix, y_delay, y_type)

    return design_matrix, y_delay, y_type


def save_to_file(X, y_delay, y_type):
    print("save to file")
    X.to_csv(f"../pickle/X_10000.csv", index=False)
    y_delay.to_csv(f"../pickle/y_delay_10000.csv", index=False)
    y_type.to_csv(f"../pickle/y_type_10000.csv", index=False)


def read_saved_files():
    print("file exists, load from file")
    design_matrix = pd.read_csv("../pickle/X_10000.csv")
    y_delay = pd.read_csv("../pickle/y_delay_10000.csv")['ArrDelay']
    y_type = pd.read_csv("../pickle/y_type_10000.csv")['DelayFactor']
    print("load complete")
    return design_matrix, y_delay, y_type


def start_train(train_path: str, weather_path: str):
    """
    Start train the data
    """
    design_matrix, y_delay, y_type = get_feature_matrix(train_path, weather_path)
    # Split to train and test
    x_train, y_train_delay, y_train_type, x_test, y_test_delay, y_test_type = split_to_train_test(design_matrix,
                                                                                                  y_delay,
                                                                                                  y_type)

    run_regression(x_test, x_train, y_test_delay, y_train_delay)
    # Get classifier model
    run_classifier(x_test, x_train, y_test_type, y_train_type)


if __name__ == '__main__':
    if is_valid_usage():
        start_train(TRAIN_DATA_FILE[sys.argv[1]], WEATHER_FILE[sys.argv[2]])
    else:
        print("Usage: python weather_preprocess.py X Y\n"
              "'X = 1k/10k/all_data' for 1k/10k/all_data flight data\n"
              "'Y = jfk' for weather with only JFK\n"
              "'Y = all_weather' for all weather data\n")

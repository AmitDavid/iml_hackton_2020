import sys
import os
from task1.src.preprocess_fllght_data import *
from task1.src.weather_preprocess import *

NUM_OF_ARGS = 3

PATH_TO_TRAIN_DATA = "../data/train_data.csv"
PATH_TO_1K_TRAIN_DATA = "../data/data1k.csv"
PATH_TO_TEST_DATA = "../data/train_data.csv"
PATH_TO_WEATHER_DATA = "../data/all_weather_data.csv"

MY_DIR = os.path.dirname(__file__)
WEATHER_FILE_PATH = os.path.join(MY_DIR, '..', 'data', 'all_weather_data.csv')
WEATHER_JFK_PATH = os.path.join(MY_DIR, '..', 'small_data', 'weather_jfk.csv')
TRAIN_DATA_FILE_PATH = os.path.join(MY_DIR, '..', 'data', 'train_data.csv')
SMALL_TRAIN_DATA_PATH = os.path.join(MY_DIR, '..', 'small_data', 'data1k.csv')
MEDIUM_TRAIN_DATA_PATH = os.path.join(MY_DIR, '..', 'small_data', 'data10k.csv')

WEATHER_FILE = {
    'all_weather': WEATHER_FILE_PATH,
    'jfk': WEATHER_JFK_PATH
}

TRAIN_DATA = {
    '1k': SMALL_TRAIN_DATA_PATH,
    '10k': MEDIUM_TRAIN_DATA_PATH,
    'all_data': TRAIN_DATA_FILE_PATH
}


def get_reg_score(y: pd.DataFrame, y_hat: pd.DataFrame) -> float:
    return -1.0


def get_class_score(y: pd.DataFrame, y_hat: pd.DataFrame) -> float:
    return sum(y == y_hat)


def is_valid_usage():
    return len(sys.argv) == NUM_OF_ARGS and (
            sys.argv[1] != 'small_data' or sys.argv[1] != 'all_data') and \
           (sys.argv[2] != 'jfk' or sys.argv[2] != 'all_weather')


def run_classifier(X_test, X_train, y_test_type, y_train_type):
    """
    Run the classifier, and print the score of the trained model
    :param X_test: Test feature matrix
    :param X_train: Train feature matrix
    :param y_test_type: Test types
    :param y_train_type: Train types
    """
    # Get reg model
    mask = X_train > 0
    class_model = get_best_class_model(X_train[mask], y_train_type[mask], X_test[mask], y_test_type[mask])
    # Run and get score
    # y_train_type_hat = class_model.predict(X_train)
    # y_test_type_hat = class_model.predict(X_test)
    # print('class train score: ', get_class_score(y_train_type, y_train_type_hat))
    # print('class test score: ', get_class_score(y_test_type, y_test_type_hat))


def run_regression(X, X_test, X_train, y_delay, y_test_delay, y_train_delay, y_type):
    """
    Run the regression, and print the score of the trained model
    :param X_test: Test feature matrix
    :param X_train: Train feature matrix
    :param y_test_type: Test types
    :param y_train_type: Train types
    """
    reg_model = get_best_reg_model(X_train, y_train_delay)
    # Run and get score
    y_train_delay_hat = reg_model.predict(X_train)
    y_test_delay_hat = reg_model.predict(X_test)
    print('reg train score: ', get_reg_score(y_train_delay, y_train_delay_hat))
    print('reg test score: ', get_reg_score(y_test_delay, y_test_delay_hat))


def get_feature_matrix():
    """
    Get the preprocessed data
    :return:  X, y_delay, y_type
    """
    df = pd.read_csv(PATH_TO_1K_TRAIN_DATA, dtype={'FlightDate': str, 'CRSDepTime': str, 'CRSArrTime': str})
    weather_df = pd.read_csv(PATH_TO_WEATHER_DATA)
    df = preprocess_weather_data(df, weather_df)
    return preprocess_flight_data(df)


def start_train():
    """
    Start train the data
    """
    X, y_delay, y_type = get_feature_matrix()
    # Split to train and test
    X_train, y_train_delay, y_train_type, X_test, y_test_delay, y_test_type = split_to_train_test(X, y_delay, y_type)
    run_regression(X, X_test, X_train, y_delay, y_test_delay, y_train_delay, y_type)
    # Get classifier model
    run_classifier(X_test, X_train, y_test_type, y_train_type)


if __name__ == '__main__':
    if is_valid_usage():
        start_train()
    else:
        print("Usage: python weather_preprocess.py X Y\n"
              "'X = 1k/10k/all_data' for 1k/10k/all_data flight data\n"
              "'Y = jfk' for weather with only JFK\n"
              "'Y = all_weather' for all weather data\n")

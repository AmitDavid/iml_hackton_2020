import sys
import os
from task1.src.preprocess_fllght_data import *
from task1.src.weather_preprocess import *
from task1.src.Classification import *
from task1.src.Regression import *

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


def get_reg_score(y: pd.DataFrame, y_hat: pd.DataFrame) -> float:
    return -1.0


def get_class_score(y: pd.DataFrame, y_hat: pd.DataFrame) -> float:
    return sum(y == y_hat)


def is_valid_usage():
    return len(sys.argv) == NUM_OF_ARGS and sys.argv[1] in TRAIN_DATA_FILE \
           and sys.argv[2] in WEATHER_FILE


def run_classifier(X_test: pa.DataFrame, X_train: pa.DataFrame, y_test_type: pa.DataFrame, y_train_type: pa.DataFrame):
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

    class_model = get_best_class_model(X_train[mask_train], y_train_type[mask_train], X_test[mask_test],
                                       y_test_type[mask_test])
    print(class_model.to_string())
    # Run and get score
    # y_train_type_hat = class_model.predict(X_train)
    # y_test_type_hat = class_model.predict(X_test)
    # print('class train score: ', get_class_score(y_train_type, y_train_type_hat))
    # print('class test score: ', get_class_score(y_test_type, y_test_type_hat))
    end = time.time()
    print("run classifier time: {}".format(end - start))


def run_regression(X_test, X_train, y_test_delay, y_train_delay):
    """
    Run the regression, and print the score of the trained model
    :param X_test: Test feature matrix
    :param X_train: Train feature matrix
    :param y_test_type: Test types
    :param y_train_type: Train types
    """
    start = time.time()

    print("run reg models")
    reg_model = get_best_reg_model(X_train, y_train_delay, X_test, y_test_delay)
    print(reg_model.to_string())

    # Run and get score
    # y_train_delay_hat = reg_model.predict(X_train)
    # y_test_delay_hat = reg_model.predict(X_test)
    # print('reg train score: ', get_reg_score(y_train_delay, y_train_delay_hat))
    # print('reg test score: ', get_reg_score(y_test_delay, y_test_delay_hat))

    end = time.time()
    print("run reg time: {}".format(end - start))


def get_feature_matrix(train_path: str, weather_path: str):
    """
    Get the preprocessed data
    :param train_path: The train data file path
    :param weather_path: The weather data file path
    :return:  X, y_delay, y_type
    """
    start = time.time()

    if os.path.isfile("../pickle/X_10000.csv"):
        print("file exists, load from file")
        X = pd.read_csv("../pickle/X_10000.csv")
        y_delay = pd.read_csv("../pickle/y_delay_10000.csv")['ArrDelay']
        y_type = pd.read_csv("../pickle/y_type_10000.csv")['DelayFactor']
        print("load complete")

    else:
        print('load data')
        df = pd.read_csv(train_path, dtype={'FlightDate': str, 'CRSDepTime': str, 'CRSArrTime': str}
                         , nrows=100000)
        print('load weather')
        weather_df = pd.read_csv(weather_path, low_memory=False)
        print('preprocess weather')
        df = preprocess_weather_data(df, weather_df)
        print('preprocess data')
        X, y_delay, y_type = preprocess_flight_data(df)
        end = time.time()
        print("load data time: {}".format(end - start))

        X.info()
        print(X.describe().to_string())
        print(X.head(50).to_string())

        print("load to file")
        X.to_csv(f"../pickle/X_10000.csv", index=False)
        y_delay.to_csv(f"../pickle/y_delay_10000.csv", index=False)
        y_type.to_csv(f"../pickle/y_type_10000.csv", index=False)

    return X, y_delay, y_type


def start_train(train_path: str, weather_path: str):
    """
    Start train the data
    """
    X, y_delay, y_type = get_feature_matrix(train_path, weather_path)
    # Split to train and test
    X_train, y_train_delay, y_train_type, X_test, y_test_delay, y_test_type = split_to_train_test(X, y_delay, y_type)

    # run_regression(X_test, X_train, y_test_delay, y_train_delay)
    # Get classifier model
    run_classifier(X_test, X_train, y_test_type, y_train_type)


if __name__ == '__main__':
    if is_valid_usage():
        start_train(TRAIN_DATA_FILE[sys.argv[1]], WEATHER_FILE[sys.argv[2]])
    else:
        print("Usage: python weather_preprocess.py X Y\n"
              "'X = 1k/10k/all_data' for 1k/10k/all_data flight data\n"
              "'Y = jfk' for weather with only JFK\n"
              "'Y = all_weather' for all weather data\n")

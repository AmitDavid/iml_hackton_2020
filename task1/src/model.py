import pickle
import sys

from task1.src.preprocess_fllght_data import *
from task1.src.weather_preprocess import *

###########################################
#        For running trained model        #
###########################################
PATH_TO_REGRESSION_MODEL = "../pickle/reg_model"
PATH_TO_CLASSIFICATION_MODEL = "../pickle/class_model"


class FlightPredictor:
    def __init__(self, path_to_weather=''):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        self.weather_df = pd.read_csv(path_to_weather)

        reg_file = open(PATH_TO_REGRESSION_MODEL, 'rb')
        self.reg_model = pickle.load(reg_file)
        reg_file.close()

        class_file = open(PATH_TO_CLASSIFICATION_MODEL, 'rb')
        self.class_model = pickle.load(class_file)
        class_file.close()

    def predict(self, X):
        """
        Recieves a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param X: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        # Preprocess data
        df = preprocess_weather_data(X, weather_df)
        new_X = preprocess_flight_data(df)

        y_delay_hat = self.reg_model.predict(X)
        y_type_hat = self.class_model.predict(X)

        cols = ['ArrDelay', 'DelayFactor']
        list_of_series = [pd.Series(y_delay_hat, index=cols), pd.Series(y_type_hat, index=cols)]
        y = pd.DataFrame(list_of_series, columns=cols)

        return y


############################################
#        For creating trained model        #
############################################
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


if __name__ == '__main__':
    if is_valid_usage():
        get_dataset(TRAIN_DATA[sys.argv[1]])
    else:
        print("Usage: python weather_preprocess.py X \n"
              "'X = 1k' for 1k flight data\n"
              "'X = 10k' for 1k flight data\n"
              "'X = all_data' for all data")

    # Load data
    df = pd.read_csv(PATH_TO_1K_TRAIN_DATA,
                     dtype={'FlightDate': str, 'CRSDepTime': str, 'CRSArrTime': str})
    weather_df = pd.read_csv(PATH_TO_WEATHER_DATA)

    # Preprocess data
    df = preprocess_weather_data(df, weather_df)
    X, y_delay, y_type = preprocess_flight_data(df)

    # Split to train and test
    X_train, y_train_delay, y_train_type, X_test,\
        y_test_delay, y_test_type = split_to_train_test(X, y_delay, y_type)

    # Get reg model
    reg_model = get_best_reg_model(X_train, y_train_delay, X_test, y_test_delay)

    # Run and get score
    y_train_delay_hat = reg_model.predict(X_train)
    y_test_delay_hat = reg_model.predict(X_test)
    print('reg train score: ', get_reg_score(y_train_delay, y_train_delay_hat))
    print('reg test score: ', get_reg_score(y_test_delay, y_test_delay_hat))

    # Get classifier model
    class_model = get_best_class_model(X_train, y_train_type, X_test, y_test_type)

    # Run and get score
    y_train_type_hat = class_model.predict(X_train)
    y_test_type_hat = class_model.predict(X_test)
    print('class train score: ', get_class_score(y_train_type, y_train_type_hat))
    print('class test score: ', get_class_score(y_test_type, y_test_type_hat))

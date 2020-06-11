import sys
from task1.src.Regression import *
from task1.src.preprocess_fllght_data import *
from task1.src.weather_preprocess import *

NUM_OF_ARGS = 3

PATH_TO_TRAIN_DATA = "../data/train_data.csv"
PATH_TO_1K_TRAIN_DATA = "../data/data1k.csv"
PATH_TO_TEST_DATA = "../data/train_data.csv"
PATH_TO_WEATHER_DATA = "../data/all_weather_data.csv"
CLASSIFICATION = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'LateAircraftDelay']

class FlightPredictor:
    def __init__(self, path_to_weather=PATH_TO_WEATHER_DATA, path_to_data=PATH_TO_TRAIN_DATA):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        raise NotImplementedError

    def predict(self, x):
        """
        Recieves a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        raise NotImplementedError


def get_class_score(y: pd.DataFrame, y_hat: pd.DataFrame) -> float:
    return sum(y == y_hat)


if __name__ == '__main__':
    if is_valid_usage():
        get_dataset(TRAIN_DATA[sys.argv[1]])
    else:
        print("Usage: python weather_preprocess.py X \n"
              "'X = 1k' for 1k flight data\n"
              "'X = 10k' for 1k flight data\n"
              "'X = all_data' for all data")

    # Load data
    df = pd.read_csv(PATH_TO_1K_TRAIN_DATA, dtype={'FlightDate': str, 'CRSDepTime': str, 'CRSArrTime': str})
    weather_df = pd.read_csv(PATH_TO_WEATHER_DATA)

    # Preprocess data
    df = preprocess_weather_data(df, weather_df)
    X, y_delay, y_type = preprocess_flight_data(df)

    # Split to train and test
    X_train, y_train_delay, y_train_type, X_test, y_test_delay, y_test_type = split_to_train_test(X, y_delay, y_type)

    # Get reg model
    reg_model = get_best_reg_model = get_model(X_train, y_train_delay, X_test, y_test_delay)

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
PATH_TO_TRAIN_DATA = "../data/train_data.csv"
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


if __name__ == '__main__':
    ml = FlightPredictor()
    # y_hat = ml.predict(PATH_TO_TEST_DATA)

import pickle

from task1.src.weather_preprocess import *
from task1.src.Classification import *

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
        df = preprocess_weather_data(X, self.weather_df)
        X = preprocess_flight_data(df)

        y_delay_hat = self.reg_model.predict(X)
        y_type_hat = self.class_model.predict(X)

        cols = ['ArrDelay', 'DelayFactor']
        list_of_series = [pd.Series(y_delay_hat, index=cols), pd.Series(y_type_hat, index=cols)]
        y = pd.DataFrame(list_of_series, columns=cols)

        return y

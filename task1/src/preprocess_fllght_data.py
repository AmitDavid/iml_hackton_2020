import pandas as pd


def preprocess_flight_data(df: pd.DataFrame):
    """
    :param df:  Pandas DataFrame contain the following:
                DayOfWeek:	                     Day of Week
                FlightDate:	                     Flight Date (yyyy-mm-dd)
                Reporting_Airline:               Unique Carrier Code
                Tail_Number:                     Tail Number
                Flight_Number_Reporting_Airline: Flight Number
                Origin:	                         Origin Airport
                OriginCityName:                  Origin Airport, City Name
                OriginState:                     Origin Airport, State Code
                Dest:                            Destination Airport
                DestCityName:                    Destination Airport, City Name
                DestState:                       Destination Airport, State Code
                CRSDepTime:                      Expected Departure Time (local time: hhmm)
                CRSArrTime:                      Expected Arrival Time (local time: hhmm)
                CRSElapsedTime:                  Expected Elapsed Time of Flight, in Minutes
                Distance:                        Distance between airports (miles)

                ArrDelay:                        Time difference (in minutes) from expected arrival time
                DelayFactor:                     Type fo delay
    :return: X - Processed data frame. y_delay - Time difference , y_type - Type fo delay
    """
    # TODO: might make them dummies as well, check if make prediction better
    # Remove OriginCityName, OriginState, DestCityName, DestState,
    del df['OriginCityName']
    del df['OriginState']
    del df['DestCityName']
    del df['DestState']

    # Get categorical features (dummies) for dayOfTheWeek, Reporting_Airline, Tail_Number
    # Flight_Number_Reporting_Airline, Origin, Dest
    df = pd.get_dummies(df, columns=['DayOfWeek', 'Reporting_Airline', 'Tail_Number',
                                     'Flight_Number_Reporting_Airline', 'Origin', 'Dest'])

    # Get hour and ten of minutes of CRSDepTime and CRSArrTime
    # If flight was at 1546.0 (15:46), save it to 154 dummy
    # If flight was at 133.0 (01:33), save it to 013 dummy (with leading zero)
    df['CRSDepTime'] = df['CRSDepTime'].str.slice(stop=-3).str.zfill(3)
    df['CRSArrTime'] = df['CRSArrTime'].str.slice(stop=-3).str.zfill(3)
    df = pd.get_dummies(df, columns=['CRSDepTime'])
    df = pd.get_dummies(df, columns=['CRSArrTime'])

    # Split dayInDate, monthInDate, yearInDate and make than dummies (yyyy-mm-dd)
    df['yearInDate'] = df['FlightDate'].str.slice(stop=4)
    df['monthInDate'] = df['FlightDate'].str.slice(start=5, stop=7)
    df['dayInDate'] = df['FlightDate'].str.slice(start=8)

    df = pd.get_dummies(df, columns=['yearInDate'])
    df = pd.get_dummies(df, columns=['monthInDate'])
    df = pd.get_dummies(df, columns=['dayInDate'])

    del df['FlightDate']

    # split ArrDelay and DelayFactor to results DataFrame
    y_delay = df['ArrDelay']
    del df['ArrDelay']

    y_type = df['DelayFactor']
    del df['DelayFactor']

    # CRSElapsedTime and Distance left unchanged
    return df, y_delay, y_type


def split_to_train_test(df, y_1, y_2, ratio=5):
    """
    Split data to train and test sets
    :param df:
    :param y_1:
    :param y_2:
    :param ratio:
    :return:
    """
    cut = int(len(df) / ratio)
    return df[cut:], y_1[cut:], y_2[cut:], df[:cut], y_1[:cut],  y_2[:cut]


if __name__ == '__main__':
    df = pd.read_csv("../data/data1k.csv", dtype={'FlightDate': str, 'CRSDepTime': str,
                                                  'CRSArrTime': str})

    X, y_delay, y_type = preprocess_flight_data(df)
    X.to_csv("results_X.csv")

    train_X, train_y_1, train_y_2, test_X, test_y_1, test_y_2 = split_to_train_test(X, y_delay, y_type)
    print(len(train_X), len(train_y_1), len(train_y_2), len(test_X), len(test_y_1), len(test_y_2))

    print('done')


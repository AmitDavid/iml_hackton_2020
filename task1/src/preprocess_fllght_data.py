import pandas as pd

CAT_COLS2 = ['CRSDepTime', 'CRSArrTime']
DATE_COLS = ['yearInDateArr', 'monthInDateArr', 'dayInDateArr', 'yearInDate', 'monthInDate', 'dayInDate']
CAT_COLS = ['DayOfWeek', 'Reporting_Airline', 'Tail_Number', 'Origin', 'Dest'] + CAT_COLS2 + DATE_COLS


def preprocess_flight_data(df: pd.DataFrame, train_data=True, empty_df=None):
    """
    :param train_data: Optional, True/False the data includes the real delays (default=True)
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
    # Convert to strings
    df['FlightDate'] = df['FlightDate'].astype(str)
    df['CRSDepTime'] = df['CRSDepTime'].astype(str)
    df['CRSArrTime'] = df['CRSArrTime'].astype(str)

    # Remove Tail_Number, OriginCityName, OriginState, DestCityName, DestState,
    del df['OriginCityName']
    del df['OriginState']
    del df['DestCityName']
    del df['DestState']

    # Get hour and ten of minutes of CRSDepTime and CRSArrTime
    # If flight was at 1546.0 (15:46), save it to 154 dummy
    # If flight was at 133.0 (01:33), save it to 013 dummy (with leading zero)
    df['CRSDepTime'] = df['CRSDepTime'].str.slice(stop=-3).str.zfill(3)
    df['CRSArrTime'] = df['CRSArrTime'].str.slice(stop=-3).str.zfill(3)

    # Get categorical features (dummies) for dayOfTheWeek, Reporting_Airline
    # Flight_Number_Reporting_Airline, Origin, Dest

    # df = pd.get_dummies(df, columns=CAT_COLS)
    # df['Flight_Number_Reporting_Airline'] = df['Flight_Number_Reporting_Airline'].astype('category')

    # df = pd.get_dummies(df, columns=['CRSDepTime'])
    # df = pd.get_dummies(df, columns=['CRSArrTime'])

    # Split dayInDate, monthInDate, yearInDate and make than dummies (yyyy-mm-dd)
    df['yearInDate'] = df['FlightDate'].str.slice(stop=4)
    df['monthInDate'] = df['FlightDate'].str.slice(start=5, stop=7)
    df['dayInDate'] = df['FlightDate'].str.slice(start=8)

    # df = pd.get_dummies(df, columns=['yearInDate'])
    # df = pd.get_dummies(df, columns=['monthInDate'])
    # df = pd.get_dummies(df, columns=['dayInDate'])

    del df['FlightDate']

    # Split dayInDate, monthInDate, yearInDate and make than dummies (yyyy-mm-dd)
    df['yearInDateArr'] = df['day_arr'].str.slice(stop=4)
    df['monthInDateArr'] = df['day_arr'].str.slice(start=5, stop=7)
    df['dayInDateArr'] = df['day_arr'].str.slice(start=8)

    # df = pd.get_dummies(df, columns=['yearInDateArr'])
    # df = pd.get_dummies(df, columns=['monthInDateArr'])
    # df = pd.get_dummies(df, columns=['dayInDateArr'])

    del df['day_arr']

    for col in CAT_COLS:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes

    if train_data:
        # split ArrDelay and DelayFactor to results DataFrame
        y_delay = df['ArrDelay']
        del df['ArrDelay']

        y_type = df['DelayFactor']
        del df['DelayFactor']

        # CRSElapsedTime and Distance left unchanged
        return df, y_delay, y_type

    else:
        # add missing features
        pd.concat([empty_df, df], axis=0).drop_duplicates().fillna(0)

        return df


def split_to_train_test(df, y_1, y_2, ratio=40000):
    """
    Split data to train and test sets
    :param df:
    :param y_1:
    :param y_2:
    :param ratio:
    :return:
    """
    cut = int(len(df) / ratio)
    return df[cut:], y_1[cut:], y_2[cut:], df[:cut], y_1[:cut], y_2[:cut]

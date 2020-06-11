import pandas as pd


def preprocess_flight_data(df: pd.DataFrame):
    '''
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
    :return: Processed data frame.
    '''
    # TODO: might make them dummies as well, check if make prediction better
    # Remove OriginCityName, OriginState, DestCityName, DestState,
    del df['OriginCityName']
    del df['OriginState']
    del df['DestCityName']
    del df['DestState']

    # Get categorical features (dummies) for dayOfTheWeek, Reporting_Airline,
    # Flight_Number_Reporting_Airline, Origin, Dest
    df = pd.get_dummies(df, columns=['DayOfWeek', 'Reporting_Airline', 'Tail_Number',
                                     'Flight_Number_Reporting_Airline', 'Origin', 'Dest'])

    # Get hour and ten of mintues of CRSDepTime and CRSArrTime
    # Tf flight was at 1546.0 (15:46), save it to 154 dummy
    df['CRSDepTime'] = df['CRSDepTime'].str.slice(stop=-3)
    df['CRSArrTime'] = df['CRSArrTime'].str.slice(stop=-3)
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
    y = pd.DataFrame()
    y['ArrDelay'] = df['ArrDelay']
    y['DelayFactor'] = df['DelayFactor']
    del df['ArrDelay']
    del df['DelayFactor']

    # CRSElapsedTime and Distance left unchanged
    return df, y

"""
if __name__ == '__main__':
    df = pd.read_csv("../data/data1k.csv", dtype={'FlightDate': str, 'CRSDepTime': str,
                                                  'CRSArrTime': str})
    X, y = preprocess_flight_data(df)
    X.to_csv("results_X.csv")
    y.to_csv("results_y.csv")
    print('done')
"""

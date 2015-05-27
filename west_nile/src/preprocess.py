import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.externals import joblib


def prepare_data():
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    # sample = pd.read_csv('../input/sampleSubmission.csv')
    weather = pd.read_csv('../input/weather.csv')
    # spray = pd.read_csv('../input/spray.csv')

    # Get labels
    labels = train.WnvPresent.values

    # Not using codesum for this benchmark
    weather = weather.drop('CodeSum', axis=1)

    # Split station 1 and 2 and join horizontally
    weather_stn1 = weather[weather['Station'] == 1]
    weather_stn2 = weather[weather['Station'] == 2]
    weather_stn1 = weather_stn1.drop('Station', axis=1)
    weather_stn2 = weather_stn2.drop('Station', axis=1)
    weather = weather_stn1.merge(weather_stn2, on='Date')

    # replace some missing values and T with -1
    weather = weather.replace('M', np.NaN)
    weather = weather.replace('-', np.NaN)
    weather = weather.replace('T', 0)
    weather = weather.replace(' T', 0)
    weather = weather.replace('  T', 0)

    # Functions to extract month and day from dataset
    # You can also use parse_dates of Pandas.
    def create_month(x):
        return x.split('-')[1]

    def create_day(x):
        return x.split('-')[2]

    train['month'] = train.Date.apply(create_month)
    train['day'] = train.Date.apply(create_day)
    test['month'] = test.Date.apply(create_month)
    test['day'] = test.Date.apply(create_day)

    # Add integer latitude/longitude columns
    train['Lat_int'] = train.Latitude.apply(int)
    train['Long_int'] = train.Longitude.apply(int)
    test['Lat_int'] = test.Latitude.apply(int)
    test['Long_int'] = test.Longitude.apply(int)

    # drop address columns
    train = train.drop(['Address', 'AddressNumberAndStreet', 'NumMosquitos', 'WnvPresent'], axis=1)
    test = test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis=1)

    # Merge with weather data
    train = train.merge(weather, on='Date')
    test = test.merge(weather, on='Date')
    train = train.drop(['Date'], axis=1)
    test = test.drop(['Date'], axis=1)

    # Convert categorical data to numbers
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train['Species'].values) + list(test['Species'].values))
    train['Species'] = lbl.transform(train['Species'].values)
    test['Species'] = lbl.transform(test['Species'].values)

    lbl.fit(list(train['Street'].values) + list(test['Street'].values))
    train['Street'] = lbl.transform(train['Street'].values)
    test['Street'] = lbl.transform(test['Street'].values)

    lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
    train['Trap'] = lbl.transform(train['Trap'].values)
    test['Trap'] = lbl.transform(test['Trap'].values)

    imp = Imputer()
    train = imp.fit_transform(train)
    test = imp.transform(test)

    joblib.dump((train, labels, test), '../data/data.pkl')


def load_data():
    return joblib.load('../data/data.pkl')

if __name__ == '__main__':
    prepare_data()

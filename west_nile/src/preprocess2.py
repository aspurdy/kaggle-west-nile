from __future__ import print_function
import numpy as np
import datetime
import csv

# Treating unspecified as PIPIENS (http://www.ajtmh.org/content/80/2/268.full)
from sklearn.externals import joblib

species_map = {'CULEX RESTUANS': "100000",
               'CULEX TERRITANS': "010000",
               'CULEX PIPIENS': "001000",
               'CULEX PIPIENS/RESTUANS': "101000",
               'CULEX ERRATICUS': "000100",
               'CULEX SALINARIUS': "000010",
               'CULEX TARSALIS': "000001",
               'UNSPECIFIED CULEX': "001000"}


def date(text):
    return datetime.datetime.strptime(text, "%Y-%m-%d").date()


def precipitation(text):
    trace = 1e-3
    text = text.strip()
    if text == "M":
        return None
    if text == "T":
        return trace
    return float(text)


def impute_missing_weather_station_values(weather):
    # Stupid simple
    for k, v in weather.items():
        if v[0] is None:
            v[0] = v[1]
        elif v[1] is None:
            v[1] = v[0]
        for k1 in v[0]:
            if v[0][k1] is None:
                v[0][k1] = v[1][k1]
        for k1 in v[1]:
            if v[1][k1] is None:
                v[1][k1] = v[0][k1]


def load_weather():
    weather = {}
    for line in csv.DictReader(open("../input/weather.csv")):
        for name, converter in {"Date": date,
                                "Tmax": float, "Tmin": float, "Tavg": float,
                                "DewPoint": float, "WetBulb": float,
                                "PrecipTotal": precipitation,
                                "Depart": float,
                                "ResultSpeed": float, "ResultDir": float, "AvgSpeed": float,
                                "StnPressure": float, "SeaLevel": float}.items():
            x = line[name].strip()
            line[name] = converter(x) if (x != "M") else None
        station = int(line["Station"]) - 1
        assert station in [0, 1]
        dt = line["Date"]
        if dt not in weather:
            weather[dt] = [None, None]
        assert weather[dt][station] is None, "duplicate weather reading {0}:{1}".format(dt, station)
        weather[dt][station] = line
    impute_missing_weather_station_values(weather)
    return weather


def load_training():
    training = []
    for line in csv.DictReader(open("../input/train.csv")):
        for name, converter in {"Date": date,
                                "Latitude": float, "Longitude": float,
                                "NumMosquitos": int, "WnvPresent": int}.items():
            line[name] = converter(line[name])
        training.append(line)
    return training


def load_testing():
    training = []
    for line in csv.DictReader(open("../input/test.csv")):
        for name, converter in {"Date": date,
                                "Latitude": float, "Longitude": float}.items():
            line[name] = converter(line[name])
        training.append(line)
    return training


def closest_station(latitude, longitude):
    # Chicago is small enough that we can treat coordinates as rectangular.
    stations = np.array([[41.995, -87.933],
                         [41.786, -87.752]])
    loc = np.array([latitude, longitude])
    deltas = stations - loc[None, :]
    dist2 = (deltas ** 2).sum(1)
    return np.argmin(dist2)


def normalize(x, mean=None, std=None):
    count = x.shape[1]
    if mean is None:
        mean = np.nanmean(x, axis=0)
    for i in range(count):
        x[np.isnan(x[:, i]), i] = mean[i]
    if std is None:
        std = np.std(x, axis=0)
    for i in range(count):
        x[:, i] = (x[:, i] - mean[i]) / std[i]
    return mean, std


def scaled_count(record):
    scale = 10.0
    if "NumMosquitos" not in record:
        # This is test data
        return 1
    return int(np.ceil(record["NumMosquitos"] / scale))


def assemble_x(base, weather):
    X = []
    for b in base:
        date = b["Date"]
        lat, longi = b["Latitude"], b["Longitude"]
        case = [date.year, date.month, date.day, lat, longi]
        # Look at a selection of past weather values
        for days_ago in [1, 2, 3, 5, 8, 12]:
            day = date - datetime.timedelta(days=days_ago)
            for obs in ["Tmax", "Tmin", "Tavg", "DewPoint", "WetBulb", "PrecipTotal", "Depart"]:
                station = closest_station(lat, longi)
                case.append(weather[day][station][obs])
        # Specify which mosquitos are present
        species_vector = [float(x) for x in species_map[b["Species"]]]
        case.extend(species_vector)
        # Weight each observation by the number of mosquitos seen. Test data
        # Doesn't have this column, so in that case use 1. This accidentally
        # Takes into account multiple entries that result from >50 mosquitos
        # on one day.
        for repeat in range(scaled_count(b)):
            X.append(case)
    X = np.asarray(X, dtype=np.float32)
    return X


def assemble_y(base):
    y = []
    for b in base:
        present = b["WnvPresent"]
        for repeat in range(scaled_count(b)):
            y.append(present)
    return np.asarray(y, dtype=np.int32)


def load_data():
    weather = load_weather()
    training = load_training()
    testing = load_testing()

    train = assemble_x(training, weather)
    mean, std = normalize(train)
    labels = assemble_y(training)

    test = assemble_x(testing, weather)
    normalize(test, mean, std)

    return train, labels, test

def pkl_data():
    (train, labels, test) = load_data()
    joblib.dump((train, labels, test), '../data/data2.pkl')

def load_pkl():
    return joblib.load('../data/data2.pkl')

if __name__ == '__main__':
    pkl_data()

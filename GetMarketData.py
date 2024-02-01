import datetime
import requests
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from full_fred.fred import Fred
import quandl

from settings.default import CURRENCIES, INDICATOR, MAPPING

EOD_KEY = "65538af05fe6a7.72416927"


def save_data(folder, name, data):
    path_directory = "data/" + folder
    path = path_directory + "/" + name + ".parquet"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path_directory)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path_directory)

    data.to_parquet(path)
    print("Write to parquet ---->", path)


def get_forex(
    pair1="EUR",
    pair2="USD",
    exchange="FOREX",
    period_end=datetime.datetime.today(),
    period_start=datetime.date(1960, 1, 1),
    period_days=100006,
    only_close=True,
):
    url = "https://eodhistoricaldata.com/api/eod/{}.{}".format(pair1, exchange)
    params = {
        "api_token": EOD_KEY,
        "fmt": "json",
        "from": period_start.strftime("%Y-%m-%d")
        if period_start
        else (period_end - datetime.timedelta(days=period_days)).strftime("%Y-%m-%d"),
        "to": period_end.strftime("%Y-%m-%d"),
    }
    resp = requests.get(url, params=params)
    df = pd.DataFrame(resp.json())
    df.rename(columns={"date": "datetime", "adjusted_close": pair1}, inplace=True)
    if only_close:
        df = df[["datetime", pair1]]

    if len(df) > 0:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def get_libor(
    period_end=datetime.datetime.today(), period_start=datetime.date(1960, 1, 1)
):
    url = "https://eodhd.com/api/eod/EURIBOR3M.MONEY"
    params = {
        "api_token": EOD_KEY,
        "fmt": "json",
        "from": period_start.strftime("%Y-%m-%d"),
        "to": period_end.strftime("%Y-%m-%d"),
    }
    resp = requests.get(url, params=params)
    df = pd.DataFrame(resp.json())
    df.rename(columns={"date": "datetime", "adjusted_close": "LIBOR3M"}, inplace=True)
    if len(df) > 0:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def get_bond(
    country="US",
    maturity="10Y",
    period_end=datetime.datetime.today(),
    period_start=datetime.date(1960, 1, 1),
):
    url = "https://eodhd.com/api/eod/{}{}.GBOND".format(country, maturity)
    params = {
        "api_token": EOD_KEY,
        "fmt": "json",
        "from": period_start.strftime("%Y-%m-%d"),
        "to": period_end.strftime("%Y-%m-%d"),
    }
    resp = requests.get(url, params=params)
    df = pd.DataFrame(resp.json())
    df["Country"] = country
    df = df[["date", "adjusted_close"]]
    df.rename(
        columns={
            "date": "datetime",
            "CountryCode": "Country",
            "adjusted_close": country + maturity,
        },
        inplace=True,
    )
    if len(df) > 0:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def get_fred_serie(
    serie, period_start="1960-01-01", period_end=datetime.datetime.today()
):
    fred = Fred("key.text")
    df = fred.get_series_df(serie).replace(".", np.nan)
    df["value"] = df["value"].astype(float)
    df = df[["date", "value"]]
    df.rename(columns={"date": "datetime", "value": serie}, inplace=True)
    if len(df) > 0:
        df["datetime"] = pd.to_datetime(df["datetime"])

    df = df[df["datetime"] > period_start]
    df = df[df["datetime"] < period_end]
    return df.dropna()


def get_quandl_futures(exchange, code, depth):
    QUANDL_API_KEY = "VeACKewaTU2bpx-yfrgF"
    quandl.ApiConfig.api_key = QUANDL_API_KEY
    name = "CHRIS/{}_{}{}".format(exchange, code, depth)
    data = quandl.get(name, start_date="1960-01-01")
    data = data.reset_index()
    data.rename(columns={"Date": "datetime", "Settle": code}, inplace=True)
    data["datetime"] = pd.to_datetime(data["datetime"])
    return data


if __name__ == "__main__":
    # for pair in CURRENCIES:
    #     fx_rate = get_forex(pair)
    #     save_data("Currencies",pair,fx_rate)

    for serie in INDICATOR:
        try:
            data = get_fred_serie(serie)
            save_data(os.path.join("Indicator", "US"), serie, data)
        except:
            pass

    # for curr in tqdm(CURRENCIES):
    #     for serie in MAPPING[curr]:
    #         try :
    #             data = get_fred_serie(serie)
    #             save_data(os.path.join("Indicator",curr),serie,data)
    #         except:
    #             pass

    # futures = [('CME','FF','1')]
    # for future in futures:
    #     exchange, code, depth = (*future,)
    #     data = get_quandl_futures(exchange,code,depth)
    #     save_data("Indicator",code,data)

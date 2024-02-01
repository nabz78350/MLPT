import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import shutil
import re
import glob
from utils import *
from settings.default import CURRENCIES
from pandas.tseries.offsets import BDay
from empyrical import sharpe_ratio
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from settings.default import MAPPING, INDICATOR
RESULTS_DIR = 'results_tuned'
VOL_LOOKBACK = 60  # for ex-ante volatility
VOL_TARGET = 0.15  # 15% volatility target
SQUEEZE_PARAM= 5 
OUTLIERS_TIME_LAG = 252


def compute_returns(price: pd.Series, day_offset: int = 1) -> pd.Series:
    """for each element of a pandas time-series price,
    calculates the returns over the past number of days
    specified by offset

    Args:
        price (pd.Series): time-series of prices
        day_offset (int, optional): number of days to calculate returns over. Defaults to 1.

    Returns:
        pd.Series: series of returns
    """
    returns = price / price.shift(day_offset) - 1.0
    return returns


def compute_volatility_daily(daily_returns,vol_lookback=VOL_LOOKBACK):
    return (
        daily_returns.ewm(span=vol_lookback, min_periods=vol_lookback)
        .std()
        .fillna(method="bfill")
    )



def compute_returns_vol_adjusted(returns, vol=pd.Series(None),annualization =252):
    """calculates volatility scaled returns for annualised VOL_TARGET of 15%
    with input of pandas series returns"""
    if not len(vol):
        vol = compute_volatility_daily(returns)
    vol = vol * np.sqrt(annualization)  
    return returns * VOL_TARGET / vol.shift(1)

def features_for_model(stock_data: pd.DataFrame) -> pd.DataFrame:

    stock_data = stock_data[
        ~stock_data["close"].isna()
        | ~stock_data["close"].isnull()
        | (stock_data["close"] > 1e-8)  # price is basically null
    ].copy()

    stock_data["srs"] = stock_data["close"]
    ewm = stock_data["srs"].ewm(halflife=15)
    means = ewm.mean()
    stds = ewm.std()
    stock_data["srs"] = np.minimum(stock_data["srs"], means + SQUEEZE_PARAM
     * stds)
    stock_data["srs"] = np.maximum(stock_data["srs"], means - SQUEEZE_PARAM
     * stds)
    stock_data["daily_returns"] = compute_returns(stock_data["srs"])
    stock_data["daily_vol"] = compute_volatility_daily(stock_data["daily_returns"])
    def return_normalised(day_offset):
        return (
            compute_returns(stock_data["srs"], day_offset)
            / stock_data["daily_vol"]
            / np.sqrt(day_offset)
        )
    stock_data["norm_daily_returns"] = return_normalised(1)
    return stock_data 

def get_profile(returns, fomc_dates,offset):
    column = returns.columns.tolist()[0]
    fomc_up = {}
    fomc_down = {}
    fomc_still = {}
    fomc_all = {}
    for date in tqdm(fomc_dates.keys()):
        change = fomc_dates[date]
        fomc_date = pd.to_datetime(date)
        start_date = fomc_date - pd.Timedelta(days=offset)
        end_date = fomc_date + pd.Timedelta(days=offset)

        relevant_data = returns.loc[start_date:end_date][[column]]
        int_days = relevant_data.index - date
        int_days = int_days.days 
        relevant_data.index = int_days
        relevant_data.loc[-offset-1] = 0
        if change == 2:

            fomc_up[date.strftime('%Y-%m-%d')] = relevant_data

        elif change ==0:
            fomc_down[date.strftime('%Y-%m-%d')] = relevant_data
        else :
            fomc_still[date.strftime('%Y-%m-%d')] = relevant_data
        fomc_all[date.strftime('%Y-%m-%d')] = relevant_data
    profile_up = pd.concat(fomc_up,axis=1).groupby(level=1,axis=1).mean().sort_index()
    profile_down = pd.concat(fomc_down,axis=1).groupby(level=1,axis=1).mean().sort_index()
    profile_still = pd.concat(fomc_still,axis=1).groupby(level=1,axis=1).mean().sort_index()
    profile_all = pd.concat(fomc_all,axis=1).groupby(level=1,axis=1).mean().sort_index()
    profile = pd.DataFrame({'up' : profile_up[column],
                          'down' : profile_down[column],
                          'still': profile_still[column],
                          'all': profile_all[column]}).ffill(limit=1)
    return profile


def plot_surface_metric(results,param1,param2,metric:str = "R2_test"):

    metric = metric
    results = results.sort_values(by = metric,ascending=False)
    max_metric = results.iloc[0].loc[metric]
    grid_x, grid_y = np.mgrid[results[param1].min():results[param1].max():100j, results[param2].min():results[param2].max():100j]
    grid_z = griddata((results[param1], results[param2]), results[metric], (grid_x, grid_y), method='cubic')
    max_point = results.iloc[0][[param1,param2,metric]].T

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(grid_x, grid_y, grid_z, cmap='RdYlGn', alpha=0.6)

    # # Highlighting the maximum point with a red dot and an arrow
    ax.scatter(max_point[param1], max_point[param2],max_metric, color='red', s=50)
    # ax.text(max_point[param1], max_point[param2], max_metric, '%s (%.2f, %.2f, %.2f)' % ('max', max_point[param1], max_point[param2], max_metric), size=10, zorder=3, color='k')
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    plt.colorbar(surface, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig("plots/"+"3D_surface_{}_{}_{}.png".format(param1,param2,metric))
    plt.show()

def plot_surface_coef(coef, param1, param2, coef_name: str = "BAA"):
    coef = coef.sort_values(by=coef_name, ascending=False)
    grid_x, grid_y = np.mgrid[coef[param1].min():coef[param1].max():100j, coef[param2].min():coef[param2].max():100j]
    grid_z = griddata((coef[param1], coef[param2]), coef[coef_name], (grid_x, grid_y), method='cubic')
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(grid_x, grid_y, grid_z, cmap='RdYlGn', alpha=0.6)
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    plt.colorbar(surface, ax=ax, label=coef_name)
    fig.tight_layout()

    if ":" in coef_name:
        coef_name = coef_name.replace(":", "")

    fig.savefig("plots/" + "3D_surface_{}_{}_{}.png".format(param1, param2, coef_name))
    plt.show()


def zscore(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z

def center(x):
    mean = x.mean(1)
    x = x.sub(mean, 0)
    return x

def compute_metrics(X,beta,Y,index,Y_raw,period):
    Y_pred = X @beta
    results =  pd.DataFrame({'pred':Y_pred.reshape(-1)},index)
    results["true"] = Y
    results = results.dropna()
    mae = mean_absolute_error(results['true'], results['pred'])
    mse = mean_squared_error(results['true'], results['pred'])
    r2 = r2_score(results['true'], results['pred'])

    results =  pd.DataFrame({'pred':Y_pred.reshape(-1)},index)
    new_index = pd.date_range(start=results.index.min(), end=results.index.max(), freq='B')
    
    if period =="all":
        ffill_lag = 10 
    else :
        ffill_lag =10

    results = results.reindex(new_index).ffill(limit=ffill_lag)
    results['true'] = Y_raw.pct_change().dropna().iloc[:,0]
    pnl = results.fillna(0).prod(axis=1)
    pnl.cumsum().plot()
    sr = sharpe_ratio(pnl)
    return sr, mae, mse, r2


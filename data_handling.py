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
import statsmodels.api as sm

RESULTS_DIR = "results_tuned"
VOL_LOOKBACK = 60  # for ex-ante volatility
VOL_TARGET = 0.15  # 15% volatility target
SQUEEZE_PARAM = 5
OUTLIERS_TIME_LAG = 252


def aggregate_data(folder):
    """
    Aggregates data from .parquet files within a specified folder.

    This function reads all .parquet files in a specified subfolder within the 'data' directory.
    It extracts a specific column (with the same name as the file) from each file, and combines
    these columns into a single pandas DataFrame. The 'datetime' column is used as the index for
    the DataFrame.

    Args:
    folder (str): The name of the subfolder within the 'data' directory from which .parquet files are read.

    Returns:
    pandas.DataFrame: A DataFrame containing combined data from all .parquet files in the specified folder.
                      Each column in the DataFrame corresponds to a .parquet file.
    """

    directory = os.path.join("data", folder)

    files = [file for file in os.listdir(directory) if file.endswith(".parquet")]

    dfs = {}
    for file in files:
        name = file.replace(".parquet", "")
        df = pd.read_parquet(directory + "/" + file)
        df.set_index("datetime", inplace=True)
        dfs[name] = df[name]

    dfs = pd.DataFrame(dfs)
    return dfs


def download_sp500():
    """
    Downloads and formats S&P 500 data from a .parquet file.

    Args:
    None

    Returns:
    pandas.DataFrame: A DataFrame with S&P 500 data. The DataFrame has a single column 'SP500'
                      and 'datetime' as the index.
    """
    sp500 = pd.read_parquet("data/SP500/SP500.parquet")
    sp500.columns = ["SP500"]
    sp500.index.names = ["datetime"]
    return sp500


def reindex_X(target: str, max_nan_pct: int = 20):
    """
    Reindexes a specified target indicator by joining it with US indicators and currency data.

    Args:
    target (str): The name of the specific target indicator to be reindexed.
    max_nan_pct (int, optional): The maximum percentage of NaN values allowed in the data.
                                 Default is 20.

    Returns:
    tuple:
        - pandas.DataFrame: A DataFrame containing the reindexed indicators with NaN values
                            forward-filled and filtered based on 'max_nan_pct'.
        - pandas.DataFrame: A DataFrame of the target indicator.
    """

    us_indicator = aggregate_data("Indicator" + "/" + "US").loc["1995":]
    specific_indicator = aggregate_data("Indicator" + "/" + target).loc["1995":]
    indicator = us_indicator.join(specific_indicator, how="outer")
    currencies = aggregate_data("Currencies")
    indicator = indicator.join(currencies[[target]])
    Y = indicator[[target]]
    indicator = indicator.drop(target, axis=1)
    indicator = indicator.ffill(limit=252)
    indicator
    percent_missing = indicator.isnull().sum() * 100 / len(indicator)
    columns = percent_missing[percent_missing < max_nan_pct].index.tolist()
    indicator = indicator[columns].dropna()
    return indicator, Y


def get_fomc_calendar():
    """
    Retrieves the schedule of FOMC (Federal Open Market Committee) meetings.

    Args:
    None

    Returns:
    list: A list of datetimes representing the scheduled FOMC meetings.
    """

    fomc_calendar = pd.read_parquet("data/FOMC/fomc_calendar.parquet").set_index(
        "datetime"
    )
    fed_meetings = fomc_calendar[fomc_calendar["unscheduled"] == False].index.tolist()
    fed_meetings
    return fed_meetings


def create_Y(Y: pd.DataFrame, dates: list, period: str = "all"):
    """
    Creates a DataFrame with percent changes in the target data for specified dates and periods.

    Args:
    Y (pd.DataFrame): The DataFrame containing the target data.
    dates (list): A list of dates for which percent changes are to be calculated.
    period (str, optional): The period relative to the dates in 'dates' list for calculating
                            percent change. Can be 'pre', 'post', or 'all'. Default is 'all'.

    Returns:
    pd.DataFrame: A DataFrame containing the percent changes for the specified dates and period.

    Raises:
    ValueError: If the 'period' is not one of 'pre', 'post', or 'all'.
    """

    if period not in ["pre", "post", "all"]:
        raise ValueError

    to_predict = pd.DataFrame(index=dates, columns=["Y"])
    for date in dates:
        try:
            if period == "pre":
                start_car = date - BDay(11)
                end_car = date
            elif period == "post":
                start_car = date + BDay(1)
                end_car = date + BDay(11)
            else:
                start_car = date - BDay(5)
                end_car = date + BDay(5)

            subset = Y.loc[start_car:end_car]
            pct_change = subset.pct_change().cumsum() * 100
            pct_change = pct_change.iloc[-1][0]
            to_predict.loc[start_car] = pct_change
        except:
            pass
    to_predict = to_predict.dropna()
    return to_predict


def data_model(
    period: str = "all",
    target: str = "EUR",
    max_nan_pct: int = 20,
    rolling_window: int = 252,
):
    """
    Constructs a data model by combining reindexed indicators and percent changes of a target with FOMC meeting dates.

    Args:
    period (str, optional): The period for calculating percent changes ('pre', 'post', or 'all'). Default is 'all'.
    target (str, optional): The target indicator for the analysis. Default is 'EUR'.
    max_nan_pct (int, optional): The maximum percentage of NaN values allowed. Default is 20.
    rolling_window (int, optional): The window size for the rolling z-score calculation. Default is 252.

    Returns:
    tuple:
        - pandas.DataFrame: A DataFrame containing the combined data, sorted by index.
        - pandas.DataFrame: A DataFrame containing the raw target data, sorted by index.
    """

    indicator, Y_raw = reindex_X(target, max_nan_pct)
    fed_meetings = get_fomc_calendar()
    Y = create_Y(Y_raw, fed_meetings, period)
    indicator = zscore(indicator, rolling_window).dropna().clip(-3, 3)
    data = Y.join(indicator).ffill(limit=252).dropna()
    return data.sort_index(), Y_raw.sort_index()


def load_data(data: pd.DataFrame, serie: str = "Y"):
    """
    Prepares and splits a DataFrame into features and target series.

    Args:
    data (pd.DataFrame): The DataFrame to be processed.
    serie (str, optional): The name of the column to be used as the target series. Default is 'Y'.

    Returns:
    tuple:
        - list: A list of feature names.
        - pandas.DataFrame: A DataFrame containing the features.
        - pandas.Series: A Series containing the target data.

    Raises:
    AssertionError: If the number of rows in the target series does not match the number of rows in the features DataFrame.
    """

    data = data.sort_index()
    X = data.drop(serie, axis=1).sort_index()
    features = X.columns.tolist()
    Y = data.loc[X.index][serie].sort_index()
    assert Y.shape[0] == X.shape[0]
    return features, X, Y


def split_data(
    data: pd.DataFrame,
    end_train: str = datetime.date(2009, 12, 31),
    end_val: str = datetime.date(2014, 12, 31),
):
    """
    Splits a DataFrame into training, validation, and test sets based on specified dates.

    Args:
    data (pd.DataFrame): The DataFrame to be split.
    end_train (str, optional): The end date for the training set. Default is December 31, 2009.
    end_val (str, optional): The end date for the validation set. Default is December 31, 2014.

    Returns:
    tuple:
        - pandas.DataFrame: The training set.
        - pandas.DataFrame: The validation set.
        - pandas.DataFrame: The test set.
    """

    train = data.loc[:end_train]
    val = data.loc[end_train:end_val]
    test = data.loc[end_val:]
    return train, val, test


def save_data_model(result_dir, target, period):
    """
    Saves the data model to specified directories and files.

    Args:
    result_dir (str): The directory where the data model will be saved.
    target (str): The target indicator for the analysis.
    period (str): The period for calculating percent changes ('pre', 'post', or 'all').

    Returns:
    None
    """

    path_target_period = os.path.join(result_dir, target, period)
    os.makedirs(path_target_period, exist_ok=True)

    data, Y_raw = data_model(period, target=target, rolling_window=252)

    _, X, Y = load_data(data, "Y")
    X_train, X_val, X_test = split_data(X)
    Y_train, Y_val, Y_test = split_data(Y)

    Y_raw.to_csv(path_target_period + "/returns.csv")
    X_train.to_csv(path_target_period + "/X_train.csv")
    X_val.to_csv(path_target_period + "/X_val.csv")
    X_test.to_csv(path_target_period + "/X_test.csv")
    Y_train.to_csv(path_target_period + "/Y_train.csv")
    Y_val.to_csv(path_target_period + "/Y_val.csv")
    Y_test.to_csv(path_target_period + "/Y_test.csv")


def aggregate_coefficients(
    results_dir: str = "results", period: str = "all", target: str = "JPY"
):
    """
    Aggregates coefficients from multiple combination of hyperpârameters subdirectories and saves them to a single CSV file.

    Args:
    results_dir (str): The directory containing subdirectories with coefficients.
    period (str): The period for calculating percent changes ('pre', 'post', or 'all').
    target (str): The target indicator for the analysis.

    Returns:
    pandas.DataFrame: A DataFrame containing aggregated coefficients.
    """
    path = os.path.join(results_dir, target, period)
    all_results = []
    for combi_name in os.listdir(path):
        if ".csv" in combi_name:
            continue
        else:
            a, b, A, B, s, R_y = reverse_combi_name(combi_name)
            combi_dir = os.path.join(path, combi_name)
            if os.path.isdir(combi_dir):
                csv_file = os.path.join(combi_dir, "coefs.csv")
                value = pd.read_csv(csv_file, index_col=0)
                value.loc["a"] = a
                value.loc["b"] = b
                value.loc["A"] = A
                value.loc["B"] = B
                value.loc["s"] = s
                value.loc["R_y"] = R_y
                all_results.append(value.T)

    all_results = pd.concat(all_results)
    all_results = all_results.reset_index(drop=True)
    all_results.to_csv(os.path.join(path, "coefs.csv"))
    return all_results


def aggregate_metrics(results_dir=RESULTS_DIR, period="all", target="JPY"):
    """
    Aggregates metrics from multiple subdirectories and saves them to a single CSV file.

    Args:
    results_dir (str): The directory containing subdirectories with metrics.
    period (str): The period for calculating percent changes ('pre', 'post', or 'all').
    target (str): The target indicator for the analysis.

    Returns:
    pandas.DataFrame: A DataFrame containing aggregated metrics.
    """

    results_dir = os.path.join(results_dir, target, period)
    all_results = []
    for combi_name in os.listdir(results_dir):
        combi_dir = os.path.join(results_dir, combi_name)
        if os.path.isdir(combi_dir):
            csv_file = os.path.join(combi_dir, "metrics.csv")
            if os.path.isfile(csv_file):
                df = pd.read_csv(csv_file, index_col=0)
                all_results.append(df)

    all_results = pd.concat(all_results)
    all_results = all_results.sort_values(by="SR_val", ascending=False)
    all_results.to_csv(os.path.join(results_dir, "metrics.csv"))
    return all_results


def get_metrics(results_dir=RESULTS_DIR, period="all", target="JPY"):
    """
    Retrieves metrics from a CSV file within a specified directory.

    Args:
    results_dir (str): The directory containing the metrics CSV file.
    period (str): The period for calculating percent changes ('pre', 'post', or 'all').
    target (str): The target indicator for the analysis.

    Returns:
    pandas.DataFrame: A DataFrame containing metrics.
    """

    results_dir = os.path.join(results_dir, target, period)
    df = pd.read_csv(os.path.join(results_dir, "metrics.csv"), index_col=0)
    return df


def get_coefs(results_dir=RESULTS_DIR, period="all", target="JPY"):
    """
    Retrieves coefficients from a CSV file within a specified directory.

    Args:
    results_dir (str): The directory containing the coefficients CSV file.
    period (str): The period for calculating percent changes ('pre', 'post', or 'all').
    target (str): The target indicator for the analysis.

    Returns:
    pandas.DataFrame: A DataFrame containing coefficients.
    """

    results_dir = os.path.join(results_dir, target, period)
    df = pd.read_csv(os.path.join(results_dir, "coefs.csv"), index_col=0)
    return df


def get_best_parameters(results: pd.DataFrame):
    """
    This function assumes that the table of all metrics accross all combinations has been sorted before according to a criterion like SR_val
    It then take the first rows, and returns the set of hypereparameters (e.g. the combination) that yielded the top score
    Args:
    results (pd.DataFrame): The DataFrame containing parameter values.

    Returns:
    tuple: A tuple containing the best parameters (a, b, A, B, s, R_y).
    """

    a, b, A, B, R_y, s = (*results.iloc[0][:6].values,)
    a = int(a)
    b = int(b)
    A = int(A)
    B = int(B)
    s = int(s)
    return a, b, A, B, s, R_y


def get_simulation_result(simulation_name, target="SP500", period="all"):
    """
    Retrieves beta values for a specific combination of hyperparams by name.

    Args:
    simulation_name (str): The name of the simulation.
    target (str, optional): The target indicator for the analysis. Default is 'SP500'.
    period (str, optional): The period for calculating percent changes ('pre', 'post', or 'all'). Default is 'all'.

    Returns:
    pandas.DataFrame: A DataFrame containing beta values for the specified simulation.
    """
    csv_coefs = os.path.join(RESULTS_DIR, target, period, simulation_name, "coefs.csv")
    beta = pd.read_csv(csv_coefs, index_col=0)
    return beta


def reverse_combi_name(input_str):
    """
    Parses a combination name string to extract parameter values.

    Args:
    input_str (str): The combination name string in the format "a={}_b={}_A={}_B={}_s={}_R_y={}".

    Returns:
    tuple: A tuple containing the extracted parameter values (a, b, A, B, s, R_y).
    """

    # Regular expression to match the pattern and capture the values
    # The updated pattern can handle varying lengths of the decimal part in R_y
    pattern = r"a=(\d+)_b=(\d+)_A=(\d+)_B=(\d+)_s=(\d+)_R_y=(\d+\.\d+)"
    # Using regular expression to find matches
    match = re.search(pattern, input_str)

    if match:
        a = int(match.group(1))
        b = int(match.group(2))
        A = int(match.group(3))
        B = int(match.group(4))
        s = int(match.group(5))
        R_y = float(match.group(6))

        return a, b, A, B, s, R_y
    else:
        return None


def get_simulation_name(a, b, A, B, s, R_y):
    """
    Formats parameter values into a simulation name string.

    Args:
    a (int): Value of 'a' parameter.
    b (int): Value of 'b' parameter.
    A (int): Value of 'A' parameter.
    B (int): Value of 'B' parameter.
    s (int): Value of 's' parameter.
    R_y (float): Value of 'R_y' parameter.

    Returns:
    str: A formatted simulation name string.
    """
    formatted_string = "a={}_b={}_A={}_B={}_s={}_R_y={}".format(a, b, A, B, s, R_y)
    return formatted_string


def recover_model(coefs, a, b, A, B, s, R_y):
    """
    Retrieves beta values for a specific combination of parameters.

    Args:
    coefs (pd.DataFrame): The DataFrame containing coefficients.
    a (int): Value of 'a' parameter.
    b (int): Value of 'b' parameter.
    A (int): Value of 'A' parameter.
    B (int): Value of 'B' parameter.
    s (int): Value of 's' parameter.
    R_y (float): Value of 'R_y' parameter.

    Returns:
    pd.DataFrame: A DataFrame containing beta values for the specified parameters.
    """

    betas = (
        coefs.query(
            "a == @a and b == @b and A == @A and B == @B and s == @s and R_y == @R_y"
        )
        .drop(["a", "b", "A", "B", "s", "R_y"], axis=1)
        .T
    )
    betas.columns = ["coef"]
    return betas


def get_prediction(X, betas, index_df):
    """
    Generates predictions using the given features and beta values.

    Args:
    X (pd.DataFrame): The DataFrame containing features.
    betas (pd.DataFrame): The DataFrame containing beta values.
    index_df (pd.Index): The index to be used for the output DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing predictions.
    """

    out = pd.DataFrame(X.values @ betas.values)
    out.index = index_df
    out.columns = ["predictions"]
    return out


def import_data(path: str, name: str):
    """
    Imports data from a CSV file and sets the index to datetime.

    Args:
    path (str): The directory path where the CSV file is located.
    name (str): The name of the CSV file (without the '.csv' extension).

    Returns:
    pd.DataFrame: A DataFrame containing the imported data with datetime as the index.
    """

    df = pd.read_csv(path + "/" + name + ".csv", index_col=0)
    df.index.names = ["datetime"]
    df.index = pd.to_datetime(df.index)
    return df


def get_x(target, period, results_dir=RESULTS_DIR):
    """
    Retrieves the feature data (X) for a specified target and period from a directory.

    Args:
    target (str): The target indicator for the analysis.
    period (str): The period for calculating percent changes ('pre', 'post', or 'all').
    results_dir (str, optional): The directory containing the data. Default is RESULTS_DIR.

    Returns:
    pd.DataFrame: A DataFrame containing the feature data with datetime as the index.
    """

    path = os.path.join(results_dir, target, period, "X_test.csv")
    X_test = pd.read_csv(path, index_col=0)
    X_test.index.names = ["datetime"]
    X_test.index = pd.to_datetime(X_test.index)
    return X_test


def get_y(target, period, results_dir=RESULTS_DIR):
    """
    Retrieves the target data (Y) for a specified target and period from a directory.

    Args:
    target (str): The target indicator for the analysis.
    period (str): The period for calculating percent changes ('pre', 'post', or 'all').
    results_dir (str, optional): The directory containing the data. Default is RESULTS_DIR.

    Returns:
    pd.DataFrame: A DataFrame containing the target data with datetime as the index.
    """

    path = os.path.join(results_dir, target, period, "returns.csv")
    Y_test = pd.read_csv(path, index_col=0)
    Y_test.index = pd.to_datetime(Y_test.index)
    return Y_test


def clean_currency_directories(currencies, results_dir):
    """
    Cleans non-CSV files from the 'all' directories of specified currencies within the results directory.

    Args:
    currencies (list): A list of currency names to clean.
    results_dir (str): The directory containing the currency data.

    Returns:
    None
    """

    for currency in currencies:
        # Construct the path to the currency's 'all' directory
        currency_dir_path = os.path.join(results_dir, currency, "all")

        # Check if the path exists and is a directory
        if os.path.exists(currency_dir_path) and os.path.isdir(currency_dir_path):
            # List all files that are not CSV files
            non_csv_files = glob.glob(os.path.join(currency_dir_path, "*"))
            non_csv_files = [f for f in non_csv_files if not f.lower().endswith(".csv")]

            # Delete all files that are not CSV files
            for file_path in non_csv_files:
                try:
                    shutil.rmtree(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        else:
            print(f"The directory {currency_dir_path} does not exist.")


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


def compute_volatility_daily(daily_returns, vol_lookback=VOL_LOOKBACK):
    """
    Computes daily volatility based on daily returns using an exponentially weighted moving average.

    Args:
    daily_returns (pd.Series): A Series containing daily returns.
    vol_lookback (int): The lookback period for computing volatility. Default is VOL_LOOKBACK.

    Returns:
    pd.Series: A Series containing daily volatility.
    """

    return (
        daily_returns.ewm(span=vol_lookback, min_periods=vol_lookback)
        .std()
        .fillna(method="bfill")
    )


def compute_returns_vol_adjusted(returns, vol=pd.Series(None), annualization=252):
    """
    Calculates volatility-scaled returns for a target annualized volatility.

    Args:
    returns (pd.Series): A Series containing returns.
    vol (pd.Series, optional): A Series containing volatility values. If not provided, it will be computed.
    annualization (int, optional): The annualization factor. Default is 252.

    Returns:
    pd.Series: A Series containing volatility-scaled returns.
    """

    """calculates volatility scaled returns for annualised VOL_TARGET of 15%
    with input of pandas series returns"""
    if not len(vol):
        vol = compute_volatility_daily(returns)
    vol = vol * np.sqrt(annualization)
    return returns * VOL_TARGET / vol.shift(1)


def features_for_profile(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares features for a stock model based on the given stock data.

    Args:
    stock_data (pd.DataFrame): A DataFrame containing stock data.

    Returns:
    pd.DataFrame: A DataFrame containing prepared features.
    """

    stock_data = stock_data[
        ~stock_data["close"].isna()
        | ~stock_data["close"].isnull()
        | (stock_data["close"] > 1e-8)  # price is basically null
    ].copy()

    stock_data["srs"] = stock_data["close"]
    ewm = stock_data["srs"].ewm(halflife=15)
    means = ewm.mean()
    stds = ewm.std()
    stock_data["srs"] = np.minimum(stock_data["srs"], means + SQUEEZE_PARAM * stds)
    stock_data["srs"] = np.maximum(stock_data["srs"], means - SQUEEZE_PARAM * stds)
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


def get_profile(returns, fomc_dates, offset):
    """
    Generates profiles of returns for different categories based on FOMC meeting dates.

    Args:
    returns (pd.DataFrame): A DataFrame containing returns data.
    fomc_dates (dict): A dictionary containing FOMC meeting dates as keys and change categories as values.
    offset (int): The number of days to consider before and after each FOMC meeting date.

    Returns:
    pd.DataFrame: A DataFrame containing profiles of returns for different categories ('up', 'down', 'still', 'all').
    """

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
        relevant_data.loc[-offset - 1] = 0
        if change == 2:
            fomc_up[date.strftime("%Y-%m-%d")] = relevant_data

        elif change == 0:
            fomc_down[date.strftime("%Y-%m-%d")] = relevant_data
        else:
            fomc_still[date.strftime("%Y-%m-%d")] = relevant_data
        fomc_all[date.strftime("%Y-%m-%d")] = relevant_data
    profile_up = pd.concat(fomc_up, axis=1).groupby(level=1, axis=1).mean().sort_index()
    profile_down = (
        pd.concat(fomc_down, axis=1).groupby(level=1, axis=1).mean().sort_index()
    )
    profile_still = (
        pd.concat(fomc_still, axis=1).groupby(level=1, axis=1).mean().sort_index()
    )
    profile_all = (
        pd.concat(fomc_all, axis=1).groupby(level=1, axis=1).mean().sort_index()
    )
    profile = pd.DataFrame(
        {
            "up": profile_up[column],
            "down": profile_down[column],
            "still": profile_still[column],
            "all": profile_all[column],
        }
    ).ffill(limit=1)
    return profile


def plot_surface_metric(results, param1, param2, metric="R2_test"):
    """
    Plots a 3D surface of a metric against two parameters.

    Args:
    results (pd.DataFrame): A DataFrame containing results data.
    param1 (str): The name of the first parameter.
    param2 (str): The name of the second parameter.
    metric (str): The name of the metric to plot. Default is "R2_test".

    Returns:
    None
    """

    metric = metric
    results = results.sort_values(by=metric, ascending=False)
    max_metric = results.iloc[0].loc[metric]
    grid_x, grid_y = np.mgrid[
        results[param1].min() : results[param1].max() : 100j,
        results[param2].min() : results[param2].max() : 100j,
    ]
    grid_z = griddata(
        (results[param1], results[param2]),
        results[metric],
        (grid_x, grid_y),
        method="cubic",
    )
    max_point = results.iloc[0][[param1, param2, metric]].T

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(grid_x, grid_y, grid_z, cmap="RdYlGn", alpha=0.6)

    # # Highlighting the maximum point with a red dot and an arrow
    ax.scatter(max_point[param1], max_point[param2], max_metric, color="red", s=50)
    # ax.text(max_point[param1], max_point[param2], max_metric, '%s (%.2f, %.2f, %.2f)' % ('max', max_point[param1], max_point[param2], max_metric), size=10, zorder=3, color='k')
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    plt.colorbar(surface, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig("plots/" + "3D_surface_{}_{}_{}.png".format(param1, param2, metric))
    plt.show()


def plot_surface_coef(coef, param1, param2, coef_name="BAA"):
    """
    Plots a 3D surface of a coefficient against two parameters.

    Args:
    coef (pd.DataFrame): A DataFrame containing coefficient data.
    param1 (str): The name of the first parameter.
    param2 (str): The name of the second parameter.
    coef_name (str): The name of the coefficient to plot. Default is "BAA".

    Returns:
    None
    """

    coef = coef.sort_values(by=coef_name, ascending=False)
    grid_x, grid_y = np.mgrid[
        coef[param1].min() : coef[param1].max() : 100j,
        coef[param2].min() : coef[param2].max() : 100j,
    ]
    grid_z = griddata(
        (coef[param1], coef[param2]), coef[coef_name], (grid_x, grid_y), method="cubic"
    )

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(grid_x, grid_y, grid_z, cmap="RdYlGn", alpha=0.6)
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    plt.colorbar(surface, ax=ax, label=coef_name)
    fig.tight_layout()

    if ":" in coef_name:
        coef_name = coef_name.replace(":", "")

    fig.savefig("plots/" + "3D_surface_{}_{}_{}.png".format(param1, param2, coef_name))
    plt.show()


def zscore(x, window):
    """
    Computes the z-score of a series using a rolling window.

    Args:
    x (pd.Series): A Series of values.
    window (int): The rolling window size.

    Returns:
    pd.Series: A Series containing z-scores.
    """

    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x - m) / s
    return z


def center(x):
    """
    Centers the values in a DataFrame along the rows (axis=1).

    Args:
    x (pd.DataFrame): A DataFrame of values.

    Returns:
    pd.DataFrame: A DataFrame with centered values.
    """

    mean = x.mean(1)
    x = x.sub(mean, 0)
    return x


def compute_metrics(X, beta, Y, index, Y_raw, period):
    """
    Computes various evaluation metrics for a model's predictions.

    Args:
    X (pd.DataFrame): A DataFrame of features.
    beta (pd.DataFrame): A DataFrame of beta coefficients.
    Y (pd.Series): A Series of target values.
    index (pd.Index): The index to be used for the results DataFrame.
    Y_raw (pd.DataFrame): A DataFrame of raw target values.
    period (str): The period for calculating metrics ('pre', 'post', or 'all').

    Returns:
    tuple: A tuple containing Sharpe Ratio, Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2).
    """

    Y_pred = X @ beta
    results = pd.DataFrame({"pred": Y_pred.reshape(-1)}, index)
    results["true"] = Y
    results = results.dropna()
    mae = mean_absolute_error(results["true"], results["pred"])
    mse = mean_squared_error(results["true"], results["pred"])
    r2 = r2_score(results["true"], results["pred"])

    results = pd.DataFrame({"pred": Y_pred.reshape(-1)}, index)
    new_index = pd.date_range(
        start=results.index.min(), end=results.index.max(), freq="B"
    )

    if period == "all":
        ffill_lag = 10
    else:
        ffill_lag = 10

    results = results.reindex(new_index).ffill(limit=ffill_lag)
    results["true"] = Y_raw.pct_change().dropna().iloc[:, 0]
    pnl = results.fillna(0).prod(axis=1)
    pnl.cumsum().plot()
    sr = sharpe_ratio(pnl)
    return sr, mae, mse, r2


def simple_ols_regression(dataframe):
    """
    Performs simple Ordinary Least Squares (OLS) regression on a DataFrame.

    Args:
    dataframe (pd.DataFrame): A DataFrame containing target and predictor variables.

    Returns:
    tuple: A tuple containing beta coefficient, t-statistic, p-value, and R-squared value.
    """

    # Define the target variable and the predictor
    Y = dataframe.iloc[:, 0]
    X = dataframe["predictions"]

    # Adding a constant to the predictor
    X = sm.add_constant(X)

    # Performing OLS regression
    model = sm.OLS(Y, X).fit()

    # Extracting required values
    beta = model.params["predictions"]
    t_stat = model.tvalues["predictions"]
    p_value = model.pvalues["predictions"]
    r_squared = model.rsquared

    return beta, t_stat, p_value, r_squared


def get_best_pnl(target, period, criterion):
    """
    Computes the PnL (Profit and Loss) and related data for the best combination of coefficients and metrics.

    Args:
    target (str): The currency target.
    period (str): The period for evaluation ('pre', 'post', or 'all').
    criterion (str): The criterion to determine the best combination ('SR_val', 'SR_test', 'R2_val', 'R2_test', etc.).

    Returns:
    tuple: A tuple containing PnL (Profit and Loss), signal (predictions), and daily returns.

    Description:
    For a given currency, this function takes all the coefficients for all combinations and all the metrics for all simulations.
    It then identifies the row in the metrics DataFrame that yielded the best Sharpe Ratio on the validation set.
    Next, it extracts the coefficients (betas) found by this combination.
    Subsequently, it computes predictions on the test set and returns the PnL, predictions (signal), and daily returns of the currency.
    """

    results = get_metrics(results_dir=RESULTS_DIR, period=period, target=target)
    coefs = get_coefs(results_dir=RESULTS_DIR, period=period, target=target)
    results = results.sort_values(by=criterion, ascending=False)
    a, b, A, B, s, R_y = (*get_best_parameters(results),)
    betas = recover_model(coefs, a, b, A, B, s, R_y)
    X_test = get_x(target, period)
    Y_test = get_y(target, period)
    predictions = get_prediction(X_test, betas, X_test.index)
    new_index = pd.date_range(
        start=predictions.index.min(), end=predictions.index.max(), freq="B"
    )
    ffill_lag = 10 if period == "all" else 5
    predictions = predictions.reindex(new_index).ffill(limit=ffill_lag)
    predictions["true"] = Y_test.pct_change()
    pnl = predictions.fillna(0).prod(axis=1)
    signal = predictions[["predictions"]]
    return pnl, signal, Y_test

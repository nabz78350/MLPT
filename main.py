from scipy import stats
import pandas as pd
import numpy as np
import os
import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from empyrical import sharpe_ratio
import random
from tqdm import tqdm
from data_handling import *
from model import *

random.seed(0)
np.random.seed(0)

RESULTS_DIR = "results_tuned"


def wrapper(a, b, A, B, s, R_y, period, target: str = "JPY", result_dir="results"):
    folder_name = get_simulation_name(a, b, A, B, s, R_y)
    path_simulation = os.path.join(result_dir, target, period, folder_name)

    if os.path.exists(os.path.join(path_simulation, "coefs.csv")):
        print(path_simulation, " already computed")
        return

    path_target = os.path.join(result_dir, target)
    os.makedirs(path_target, exist_ok=True)

    path_target_period = os.path.join(result_dir, target, period)
    os.makedirs(path_target_period, exist_ok=True)
    X_train = import_data(path_target_period, "X_train")
    X_val = import_data(path_target_period, "X_val")
    X_test = import_data(path_target_period, "X_test")
    Y_train = import_data(path_target_period, "Y_train")
    Y_val = import_data(path_target_period, "Y_val")
    Y_test = import_data(path_target_period, "Y_test")
    Y_raw = import_data(path_target_period, "returns")
    features = X_train.columns.tolist()

    beta = compute_one_dataset(X_train.values, Y_train.values, a, b, A, B, R_y, s)
    sr_test, mae_test, mse_test, r2_test = compute_metrics(
        X=X_test.values,
        beta=beta,
        Y=Y_test.values,
        index=X_test.index.to_list(),
        Y_raw=Y_raw,
        period=period,
    )

    sr_val, mae_val, mse_val, r2_val = compute_metrics(
        X=X_val.values,
        beta=beta,
        Y=Y_val.values,
        index=X_val.index.tolist(),
        Y_raw=Y_raw,
        period=period,
    )
    sr_train, mae_train, mse_train, r2_train = compute_metrics(
        X=X_train.values,
        beta=beta,
        Y=Y_train.values,
        index=X_train.index.tolist(),
        Y_raw=Y_raw,
        period=period,
    )
    new_row = pd.DataFrame(
        [
            {
                "a": a,
                "b": b,
                "A": A,
                "B": B,
                "R_y": R_y,
                "s": s,
                "mae_test": mae_test,
                "mse_test": mse_test,
                "R2_test": r2_test,
                "SR_test": sr_test,
                "mae_val": mae_val,
                "mse_val": mse_val,
                "R2_val": r2_val,
                "SR_val": sr_val,
                "mae_train": mae_train,
                "mse_train": mse_train,
                "R2_train": r2_train,
                "SR_train": sr_train,
            }
        ]
    )

    beta = pd.DataFrame(beta, index=features, columns=["coef"])
    save_metrics(a, b, A, B, s, R_y, beta, new_row, path_target_period)
    print(path_simulation, " done ")
    return new_row, beta


def save_metrics(a, b, A, B, s, R_y, beta, results, results_dir="results"):
    folder_name = get_simulation_name(a, b, A, B, s, R_y)
    path_simulation = os.path.join(results_dir, folder_name)
    os.makedirs(path_simulation, exist_ok=True)
    path_coef = os.path.join(path_simulation, "coefs.csv")
    path_metrics = os.path.join(path_simulation, "metrics.csv")
    beta.to_csv(path_coef)
    results.to_csv(path_metrics)


def grid_search(
    a_list,
    b_list,
    A_list,
    B_list,
    s_list,
    R_y_list,
    periods,
    targets,
    results_dir="results",
):
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    else:
        pass
    Parallel(n_jobs=5)(
        delayed(wrapper)(a, b, A, B, s, R_y, period, target, results_dir)
        for a in a_list
        for b in b_list
        for A in A_list
        for B in B_list
        for s in s_list
        for R_y in R_y_list
        for period in periods
        for target in targets
    )


if __name__ == "__main__":
    targets = CURRENCIES

    for curr in tqdm(targets):
        save_data_model(RESULTS_DIR, curr, "all")
    a_list = [1, 2, 3]
    b_list = [1, 2, 3]
    A_list = [1, 2, 3]
    B_list = [1, 2, 3]
    s_list = [10, 20, 30]
    R_y_list = [0.25, 0.5, 0.75]
    periods = ["all"]
    grid_search(
        a_list=a_list,
        b_list=b_list,
        A_list=A_list,
        B_list=B_list,
        s_list=s_list,
        R_y_list=R_y_list,
        periods=periods,
        targets=targets,
        results_dir=RESULTS_DIR,
    )
    for curr in tqdm(targets):
        metrics = aggregate_metrics(RESULTS_DIR, period="all", target=curr)
        aggregate_coefficients(RESULTS_DIR, period="all", target=curr)

# Import Libraries
import pandas as pd
import numpy as np
import sklearn as sk
import PreProcessor as pp  # Self Made Module
import BackPropogationMLR as bpm


def begin():
    # Import Dataset
    dataset = pd.read_csv('CSV/CTG.csv')

    # Pre-processing data
    dataset = pp.clean_nan(dataset)
    print(dataset.shape)
    X, y = pp.split_iv_dv(dataset=dataset, exclude=(0, 1, 2, 39))

    print(pp.get_balance(y))

    # Making dataset imbalanced
    from imblearn.datasets import make_imbalance
    X_resampled, y_resampled = make_imbalance(X, y, ratio=0.05, min_c_=3, random_state=0)

    print('Synthetic generation:\n', pp.get_balance(y_resampled))

    X_csv = pd.DataFrame(X_resampled)
    y_csv = pd.DataFrame(y_resampled)

    dataframe = pd.concat((X_csv, y_csv), axis=1)
    dataframe.columns = ['b', 'e', 'LBE', 'LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'DL', 'DS', 'DP', 'DR', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP', 'CLASS', 'NSP']
    dataframe.to_csv('CTG_imb.csv', index=False)
    return 0
    # Generate Synthetic vals
    # from imblearn.over_sampling import SMOTE
    # smote = SMOTE(kind='regular')
    # X_sm, y_sm = smote.fit_sample(X_resampled, y_resampled)
    # print('Synthetic generation:\n', pp.get_balance(y_sm))
begin()

# def process_data():
dataset = pd.read_csv('CTG_imb.csv')
X_lr, y_lr, y_pred = bpm.fit_data(dataset)
print(bpm.summary(X=X_lr, y=y_lr))

# RFG
import randomfg as rfg
rfg.rfg(dataset, 10)

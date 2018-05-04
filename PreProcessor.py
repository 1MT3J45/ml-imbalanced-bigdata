import pandas as pd
import numpy as np

def clean_nan(dataset):
    dataset = dataset.dropna()
    return dataset


def split_iv_dv(dataset, exclude):
    IV, DV = 0, 0

    x = list(exclude)
    size = dataset.columns.__len__()
    runner = list(range(0,size))
    iv_cat = [i for i in runner if i not in x]

    if type(dataset) is type(np.array([])):
        print('Splitting', type(dataset))
        IV = dataset[0:, iv_cat]
        DV = dataset[0:, -1]
        #return IV, DV
    elif type(dataset) is type(pd.DataFrame()):
        print('Splitting', type(dataset))
        IV = dataset.iloc[0:, iv_cat]
        DV = dataset.iloc[0:, -1]
        return IV, DV
    else:
        print('Only numpy.NDArray & Panel.Dataframe supported')
        exit(0)


def get_balance(y):
    ibl = list()
    for idx in range(1, len(list(set(y)))+1):
        print('Entries with %d' % idx, 'is', [i for i in y].count(idx))
        ibl.append([i for i in y].count(idx))
    return ibl
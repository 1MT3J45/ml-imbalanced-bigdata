# MULTILINEAR REGRESSION

# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING DATASET
def fit_data(data):
    dataset = data# pd.read_csv("")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    print("FIRST X ___\n", X.view())
    print("FIRST y ___\n", y.view())

    # Splitting Data into Training & Testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Fitting Multiple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set
    y_pred = regressor.predict(X_test)
    return X, y, y_pred


def summary(X, y):
    # Building Optimal Model for Backward Elimination Model
    import statsmodels.formula.api as sm

    # Add unit column for constant value to sustain the in records
    #X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)

    X_opt = X[:, range(0, X.shape[1])]
    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    print(regressor_OLS.summary())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("CSV/cereal.csv", delimiter='\t')
X = dataset.iloc[0:, :-1].values
y = dataset.iloc[0:, -1].values

# Splitting Data into Training & Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

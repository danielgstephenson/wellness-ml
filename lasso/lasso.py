from math import sqrt

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import pandas as pd
from pathlib import Path
import os

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
print(os.getcwd())

X = pd.read_csv('../data/clean/controls_scaled.csv')

outcomes = pd.read_csv('../data/clean/outcomes.csv')
y = outcomes['spend_0816_0717']

print('X',X.shape)
print('y',y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
mean_y_test = np.mean(y_test)
SST = sum((y_test-mean_y_test)**2)

lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)
print('alpha',lasso_cv.alpha_)

print('')
print('Excluded:')
print(X.columns[lasso_cv.coef_ == 0])

print('')
print('Included:')
print(X.columns[lasso_cv.coef_ != 0])
print('')

predictions = lasso_cv.predict(X_test)
residuals = y_test - predictions
SSR = sum(residuals**2)
R2 = 1 - SSR/SST
print(f'R2: {R2:.4f}')




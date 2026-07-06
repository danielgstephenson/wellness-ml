import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
import pandas as pd
from pathlib import Path
import os

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
print(os.getcwd())

# The lasso regression often seems to do better without the interactions

X = pd.read_csv('../data/interaction/controls_with_interactions.csv')
# X = pd.read_csv('../data/clean/controls_scaled.csv')

outcomes = pd.read_csv('../data/clean/outcomes.csv')
y = outcomes['spend_0816_0717']

treatment = pd.read_csv('../data/clean/treatment.csv')
z = treatment['treat']

print('X',X.shape)
print('y',y.shape)
print('z',z.shape)

X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=0.5)
mean_test = np.mean(y_test)
SST = sum((y_test-mean_test)**2)

alphas = np.logspace(1, 3, 100)
lasso_cv = LassoCV(alphas=alphas, cv=10)
lasso_cv.fit(X_train, y_train)
print(f'alpha: {lasso_cv.alpha_:0.4f}')
print(f'Excluded: {sum(lasso_cv.coef_ == 0)}')
print(f'Included: {sum(lasso_cv.coef_ != 0)}')
u = y_test - lasso_cv.predict(X_test)
SSR = sum(u**2)
R2 = 1 - SSR/SST
print(f'R2: {R2:.4f}')

# We should use L1 regularization in the neural network.

# print('')
# print('Excluded:')
# print(X.columns[lasso_cv.coef_ == 0])

# print('')
# print('Included:')
# print(X.columns[lasso_cv.coef_ != 0])
# print('')




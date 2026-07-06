import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import itertools

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
print(os.getcwd())

X = pd.read_csv('../data/interaction/controls_with_interactions.csv')

outcomes = pd.read_csv('../data/clean/outcomes.csv')
y = outcomes['spend_0816_0717']

treatment = pd.read_csv('../data/clean/treatment.csv')
z = treatment['treat']

print('X',X.shape)
print('y',y.shape)

X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=0.5)
mean_test = np.mean(y_test)
SST = sum((y_test-mean_test)**2)

alphas = np.logspace(1, 3, 100)
kf = KFold(n_splits=10)
cv_errors = []

for alpha in alphas:
    fold_errors = []
    print('test',alpha)
    for fold_idx, val_idx in kf.split(X_train):
        X_fold, X_val = X_train.iloc[fold_idx,:], X_train.iloc[val_idx,:]
        y_fold, y_val = y_train.iloc[fold_idx], y_train.iloc[val_idx]
        model = Lasso(alpha=alpha, max_iter=5000, tol=1e-5)
        model.fit(X_fold, y_fold)
        preds = model.predict(X_val)
        fold_errors.append(mean_squared_error(y_val, preds))
    cv_errors.append(np.mean(fold_errors))

alpha = alphas[cv_errors==min(cv_errors)][0]
print('alpha',alpha)

# Plotting the curve
plt.figure(figsize=(8, 5))
plt.semilogx(alphas, cv_errors, marker='o', linestyle='-', markersize=2)
# plt.axvline(x=
plt.xlabel('Alpha (Log Scale)')
plt.ylabel('Average MSE')
plt.title('Validation Curve (Manual CV)')
plt.grid(True)
plt.savefig('ValidationCurve.pdf', bbox_inches='tight') 
plt.close() 
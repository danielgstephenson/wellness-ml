import pandas as pd
from pathlib import Path
import os
import itertools

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
print(os.getcwd())

X = pd.read_csv('../clean/controls_scaled.csv')

combinations = list(itertools.combinations(X.columns,2))
for col1, col2 in combinations:
    interaction_name = f'{col1}_{col2}'
    interaction = pd.Series(X[col1] * X[col2], name=interaction_name)
    X = pd.concat([X, interaction], axis=1)

X.to_csv('controls_with_interactions.csv',index=False)
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
import os

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

scaler = StandardScaler()
scaler.set_output(transform="pandas")

controls = pd.read_csv('controls.csv')
controls_scaled = pd.DataFrame(scaler.fit_transform(controls))
controls_scaled.to_csv('controls_scaled.csv',index=False)




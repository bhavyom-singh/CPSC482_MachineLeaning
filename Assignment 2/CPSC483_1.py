import pandas as pd
import numpy as np

all_data = pd.read_csv('Data\Data1.csv')

all_data = all_data.iloc[:,0:4]

print(all_data.head())
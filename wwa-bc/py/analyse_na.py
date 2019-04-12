import pandas as pd

df = pd.read_csv('../data/bc-data.csv', header=None)
nulls = df.isnull().any().any()
if nulls:
    print("There are 'null' values.")
else:
    print("There are NO 'null' values.")

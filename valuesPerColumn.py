import pandas as pd
from labels import data_headers

train_df = pd.read_csv("Datasets/NSL-KDD/KDDTrain+.txt", header=None, names=data_headers)
max_df = train_df.max(axis=0)
min_df = train_df.min(axis=0)
print(max_df)
print(min_df)
print(max_df[0])
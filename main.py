import csv
import shutil
import gzip
import os
import pandas as pd
import time
import numpy as np
from dask import dataframe as dd

file_name = "part-r-0009"

def set_category(df):
    df.loc[df['source'].str.contains("Sports", case=False), "genre"] = "Sports"
    df.loc[df['source'].str.contains("Business", case=False), "genre"] = "Business"
    df.loc[df['source'].str.contains("Health", case=False), "genre"] = "Health"
    df.loc[df['source'].str.contains("Finance", case=False), "genre"] = "Finance"
    df.loc[df['source'].str.contains("Politics", case=False), "genre"] = "Politics"
    df = df.dropna()
    return df

start = time.time()
sample_df = dd.read_csv(
    f"./dataset/{file_name}*.csv",
    delimiter=',',
    header = None,
    names=["type", "np1", "np2", "context", "source", "category", "loc", "time"],
    dtype={'loc': 'object','type': 'object'},
    engine='python',
    on_bad_lines='skip')
print("Get tail with dask: ",(time.time()-start),"sec")

start = time.time()
sample_df["np"] = sample_df["np1"].astype(str) + " " + sample_df["np2"].astype(str)
df = sample_df.map_partitions(set_category)
df = df[["np", "genre"]]
df = df.compute(num_workers=10)
df = df[df['np'].str.len() >= 20]
df = df.drop_duplicates()
print(df["genre"].value_counts())
print(df.shape)
print("Get data with dask: ",(time.time()-start),"sec")

df.to_csv('output_part9.csv', index=False)

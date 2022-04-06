import csv
import shutil
import gzip
import os
import pandas as pd
import time
import numpy as np
import gzip
import multiprocessing as mp

start = time.time()

def gz_extract(file_name):
    with gzip.open(f"../nguthidi/single/{file_name}.gz", 'rb') as f_in:
        with open(f"./dataset/{file_name}", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    with open(f"./dataset/{file_name}", 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split('\t') for line in stripped if line)
        with open(f"./dataset/{file_name}.csv", 'w', newline='') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(lines)
        os.remove(f"./dataset/{file_name}")

def fanout_unziptar(path):
    """create pool to extract all"""
    my_files = []
    for root, dirs, files in os.walk(path):
        for i in files:
            if i.endswith(".gz"):
                my_files.append(i.split(".")[0])

    pool = mp.Pool(10) # number of workers
    pool.map(gz_extract, my_files, chunksize=1)
    pool.close()


fanout_unziptar("../nguthidi/single/")
print("Extract and convert file: ",(time.time()-start),"sec")
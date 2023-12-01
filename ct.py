import numpy as np
import pandas as pd
from statistics import stdev, mode, median

dataset_path = "./dataset/adult.csv"

df = pd.read_csv(dataset_path, encoding= 'unicode_escape')

central_tendency = dict()

central_tendency["age"] = dict()
central_tendency["age"]["maen"] = df["age"].mean()
central_tendency["age"]["min"] = df["age"].min()
central_tendency["age"]["max"] = df["age"].max()
central_tendency["age"]["mode"] = mode(df["age"])
central_tendency["age"]["median"] = median(df["age"])
central_tendency["age"]["variance"] = df["age"].var()
central_tendency["age"]["standard_deviation"] = stdev(df["age"])


central_tendency["hours.per.week"] = dict()
central_tendency["hours.per.week"]["mean"] = df["hours.per.week"].mean()
central_tendency["hours.per.week"]["min"] = df["hours.per.week"].min()
central_tendency["hours.per.week"]["max"] = df["hours.per.week"].max()
central_tendency["hours.per.week"]["mode"] = mode(df["hours.per.week"])
central_tendency["hours.per.week"]["median"] = median(df["hours.per.week"])
central_tendency["hours.per.week"]["variance"] = df["hours.per.week"].var()
central_tendency["hours.per.week"]["standard_deviation"] = stdev(df["hours.per.week"])


central_tendency["capital.gain"] = dict()
central_tendency["capital.gain"]["mean"] = df["capital.gain"].mean()
central_tendency["capital.gain"]["min"] = df["capital.gain"].min()
central_tendency["capital.gain"]["max"] = df["capital.gain"].max()
central_tendency["capital.gain"]["mode"] = mode(df["capital.gain"])
central_tendency["capital.gain"]["median"] = median(df["capital.gain"])
central_tendency["capital.gain"]["variance"] = df["capital.gain"].var()
central_tendency["capital.gain"]["standard_deviation"] = stdev(df["capital.gain"])


central_tendency["capital.loss"] = dict()
central_tendency["capital.loss"]["mean"] = df["capital.loss"].mean()
central_tendency["capital.loss"]["min"] = df["capital.loss"].min()
central_tendency["capital.loss"]["max"] = df["capital.loss"].max()
central_tendency["capital.loss"]["mode"] = mode(df["capital.loss"])
central_tendency["capital.loss"]["median"] = median(df["capital.loss"])
central_tendency["capital.loss"]["variance"] = df["capital.loss"].var()
central_tendency["capital.loss"]["standard_deviation"] = stdev(df["capital.loss"])


central_tendency["education.num"] = dict()
central_tendency["education.num"]["mean"] = df["education.num"].mean()
central_tendency["education.num"]["min"] = df["education.num"].min()
central_tendency["education.num"]["max"] = df["education.num"].max()
central_tendency["education.num"]["mode"] = mode(df["education.num"])
central_tendency["education.num"]["median"] = median(df["education.num"])
central_tendency["education.num"]["variance"] = df["education.num"].var()
central_tendency["education.num"]["standard_deviation"] = stdev(df["education.num"])


central_tendency["fnlwgt"] = dict()
central_tendency["fnlwgt"]["mean"] = df["fnlwgt"].mean()
central_tendency["fnlwgt"]["min"] = df["fnlwgt"].min()
central_tendency["fnlwgt"]["max"] = df["fnlwgt"].max()
central_tendency["fnlwgt"]["mode"] = mode(df["fnlwgt"])
central_tendency["fnlwgt"]["median"] = median(df["fnlwgt"])
central_tendency["fnlwgt"]["variance"] = df["fnlwgt"].var()
central_tendency["fnlwgt"]["standard_deviation"] = stdev(df["fnlwgt"])



for k in central_tendency:
    print(f"{k}:")
    for ct in central_tendency[k]:
        print(f"\t{ct}: {central_tendency[k][ct]}")
    print()

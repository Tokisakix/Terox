import pandas as pd
from typing import List

PATH    = "data.csv"
CLASSES = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}

def getIrisDataSet() -> List:
    raw_data = pd.read_csv(PATH).to_numpy()
    data = []
    for row in raw_data:
        feature = (row[:4] - 5) / 2
        label   = CLASSES[row[4]]
        data.append((feature, label))
    return data
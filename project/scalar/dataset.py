from typing import List
from random import random

DATA = []

def getDataSet(num:int) -> List:
    for _ in range(num):
        x = 0.0
        y = 0.0
        if x == 0.0 or y == 0.0:
            x = random() - 0.5
            y = random() - 0.5
        if x * x > y:
            label = 0
        else:
            label = 1
        DATA.append(([x, y], label))
    return DATA
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
        if x > 0 and y > 0:
            label = 0
        if x < 0 and y > 0:
            label = 1
        if x < 0 and y < 0:
            label = 2
        if x > 0 and y < 0:
            label = 3
        DATA.append(([x, y], label))
    return DATA
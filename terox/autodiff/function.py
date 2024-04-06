import math

def add(a:float, b:float) -> float:
    res = a + b
    return res

def sub(a:float, b:float) -> float:
    res = a - b
    return res

def mul(a:float, b:float) -> float:
    res = a * b
    return res

def div(a:float, b:float) -> float:
    res = a / b
    return res

def inv(a:float) -> float:
    res = 1.0 / a
    return res

def neg(a:float) -> float:
    res = -a
    return res

def max(a:float, b:float) -> float:
    res = a if a > b else b
    return res

def min(a:float, b:float) -> float:
    res = a if a < b else b
    return res

def eq(a:float, b:float) -> float:
    res = 1.0 if a == b else 0.0
    return res

def lt(a:float, b:float) -> float:
    res = 1.0 if a < b else 0.0
    return res

def gt(a:float, b:float) -> float:
    res = 1.0 if a > b else 0.0
    return res

def abs(a:float) -> float:
    res = a if a > 0.0 else -a
    return res

def exp(a:float) -> float:
    res = math.exp(a)
    return res

def log(a:float) -> float:
    res = math.log(a)
    return res

def relu(a:float) -> float:
    res = a if a > 0.0 else 0.0
    return res
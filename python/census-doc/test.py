import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext


# Set the desired precision
getcontext().prec = 50  # You can adjust this value as needed

def equation(x, n):
    x = Decimal(x)
    n = Decimal(n)
    exp_n = Decimal(np.exp(n))
    return (exp_n * x) / (1 + (exp_n - 1) * x)

y = equation(0.5, 50)
print(y)
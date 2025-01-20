import numpy as np
from decimal import Decimal, getcontext
from scipy.integrate import simpson

def big_equation(x, n):
    values = np.array([])
    n = Decimal(n)
    for xv in x:
        xv = Decimal(xv)
        exp_n = n.exp()
        improve = (exp_n * xv) / (1 + (exp_n - 1) * xv)
        values = np.append(values, float(improve))
    return values

getcontext().prec = 100

# Generate a range of x values
x = np.linspace(0, 1, 400)
epsilon_zero = big_equation(x, 0)
epsilon_infinite = np.full(400, 1.0)
epsilon_zero_area = simpson(epsilon_zero, x=x)
print(f"Area under the curve for epsilon = 0: {epsilon_zero_area}")
area_simpson = simpson(epsilon_infinite, x=x)
print(f"Area under the curve for epsilon = infinite: {area_simpson}")

# Let's compute "privacy loss" as the area under curve stuff for example epsilons
examples = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
for epsilon in examples:
    y = big_equation(x, epsilon)
    area_simpson = simpson(y, x=x)
    privacy_loss = (area_simpson - epsilon_zero_area) / (1 - epsilon_zero_area)
    print(f"Privacy loss for epsilon = {epsilon}: {privacy_loss:.5f}")

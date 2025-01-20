import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from scipy.integrate import simpson

# Define the equation as a function
def equation(x, n):
    return (np.exp(n) * x) / (1 + (np.exp(n) - 1) * x)

def big_equation(x, n):
    x = Decimal(x)
    n = Decimal(n)
    exp_n = Decimal(np.exp(n))
    return (exp_n * x) / (1 + (exp_n - 1) * x)

# Generate a range of x values
x = np.linspace(0, 1, 400)
epsilon_zero = equation(x, 0)
epsilon_infinite = np.full(400, 1.0)
epsilon_zero_area = simpson(epsilon_zero, x=x)
print(f"Area under the curve for epsilon = 0: {epsilon_zero_area}")
area_simpson = simpson(epsilon_infinite, x=x)
print(f"Area under the curve for epsilon = infinite: {area_simpson}")

# Let's compute "privacy loss" as the area under curve stuff for example epsilons
examples = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
for epsilon in examples:
    y = equation(x, epsilon)
    area_simpson = simpson(y, x=x)
    privacy_loss = (area_simpson - epsilon_zero_area) / (1 - epsilon_zero_area)
    print(f"Privacy loss for epsilon = {epsilon}: {privacy_loss:.5f}")

# Values of n and their corresponding labels
n_values = [0, 1, 4, 7, 10, 50]
v_align = ['top', 'top', 'top', 'top', 'top', 'bottom']
v_pads = [-0.01, -0.01, -0.01, -0.0001, -0.0001, 0]
h_align = ['left', 'left', 'left', 'left', 'left', 'left']
h_pads = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
labels = [
    'Epsilon = 0, perfect privacy',
    'Epsilon = 1, strong privacy',
    'Epsilon = 4, Dwork\'s "privacy theatre"',
    'Epsilon = 7, Desfontaines\' "empirical testing limit"',
    'Epsilon = 10, Williams\' "smoke and mirrors"',
    'Epsilon = 50, Census 2020 Decennial Release'
]

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the curves on the left (linear scale)
for n, label, v, h, vp, hp in zip(n_values, labels, v_align, h_align, v_pads, h_pads):
    y = equation(x, n)
    line, = ax1.plot(x, y, label=label)
    if n in [0, 1, 4]:
        y_value = equation(0.5, n)  # x = 0.5
        ax1.plot(0.5, y_value, 'o', color=line.get_color())  # Solid circle with same color as line
        ax1.text(0.5+hp, y_value+vp, f'{y_value:.2f}', fontsize=12, verticalalignment=v, horizontalalignment=h, color='black')

# Plot the curves on the right with y-axis range from 0.99 to 1.0004
for n, label, v, h, vp, hp in zip(n_values, labels, v_align, h_align, v_pads, h_pads):
    y = equation(x, n)
    line, = ax2.plot(x, y, label=label)
    if n in [7, 10, 50]:
        y_value = equation(0.5, n)  # x = 0.5
        ax2.plot(0.5, y_value, 'o', color=line.get_color())  # Solid circle with same color as line
        if n == 50:
            val = big_equation(0.5, n)
            ax2.text(0.5+hp, y_value+vp, f'{val:.22f}', fontsize=12, verticalalignment=v, horizontalalignment=h, color='black')
            val = big_equation(0.000001, n)
            ax2.text(0.000001+hp, y_value+vp, f'{val:.16f}', fontsize=12, verticalalignment=v, horizontalalignment=h, color='black')
            ax2.plot(0.000001, val, 'o', color=line.get_color())  # Solid circle with same color as line
        else:
            ax2.text(0.5+hp, y_value+vp, f'{y_value:.5f}', fontsize=12, verticalalignment=v, horizontalalignment=h, color='black')

# Set labels and legend for the left plot
ax1.set_xlabel('Prior suspicion', fontsize=14)
ax1.set_ylabel('Worst-case updated suspicion', fontsize=14)
ax1.legend()
ax1.grid(True)

# Set labels and legend for the right plot
ax2.set_xlabel('Prior suspicion', fontsize=14)
ax2.set_ylabel('Worst-case updated suspicion', fontsize=14)
ax2.set_ylim(0.99, 1.0004)
ax2.legend()
ax2.grid(True)

# Increase the font size of the tick labels
ax1.tick_params(axis='both', which='major', labelsize=12)
ax2.tick_params(axis='both', which='major', labelsize=12)

# Save the plot as both PDF and PNG
plt.tight_layout()
plt.savefig('epsilon-plot.pdf')
plt.savefig('epsilon-plot.png')

# Optionally, you can show the plot
plt.show()
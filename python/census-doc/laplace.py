import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace

epsilons = [0.1, 0.5, 1, 2, 5, 10]

# Generate x values for plotting
x = np.linspace(-3, 3, 1000)


print(f'Probability that a sample from Laplace distribution is between -0.5 and 0.5:')

# Plot the Laplace distributions
plt.figure(figsize=(12, 8))
for epsilon in epsilons:
    b = 1/epsilon
    # Create a Laplace distribution with scale parameter b and loc=0
    laplace_dist = laplace(loc=0, scale=b)
    
    # Compute the PDF (Probability Density Function)
    pdf = laplace_dist.pdf(x)
    
    # Plot the PDF
    plt.plot(x, pdf, label=f'epsilon = {epsilon}')
    
    # Compute the probability that a sample is between -0.5 and 0.5
    prob = laplace_dist.cdf(0.5) - laplace_dist.cdf(-0.5)
    print(f'    epsilon = {epsilon}: {prob:.5f}')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Laplace Distributions with Different Scale Parameters (b)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
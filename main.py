import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate random data
np.random.seed(0)  # For reproducibility
x = np.random.rand(100) * 10  # 100 random points between 0 and 10
y = 2.5 * x + np.random.randn(100) * 2  # Linear relation with some noise

# Step 2: Calculate the regression line
# Using the formula: y = mx + b
# where m = (N * Σ(xy) - Σx * Σy) / (N * Σ(x^2) - (Σx)^2)
# and b = (Σy - m * Σx) / N

N = len(x)
m = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x ** 2) - (np.sum(x)) ** 2)
b = (np.sum(y) - m * np.sum(x)) / N

# Calculate the regression line
regression_line = m * x + b

# Step 3: Print data and regression line
print("Generated Data Points:")
for xi, yi in zip(x, y):
    print(f"x: {xi:.2f}, y: {yi:.2f}")

print("\nRegression Line:")
print(f"y = {m:.2f}x + {b:.2f}")

# Plotting the data and regression line
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, regression_line, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Example')
plt.legend()
plt.show()

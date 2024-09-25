import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Generate random data
np.random.seed(0)  # For reproducibility
x = np.random.rand(100, 1) * 10  # 100 random points between 0 and 10, reshaped for sklearn
y = 2.5 * x.flatten() + np.random.randn(100) * 2  # Linear relation with some noise

# Step 2: Create and fit the model
model = LinearRegression()
model.fit(x, y)

# Step 3: Get the slope (m) and intercept (b)
m = model.coef_[0]
b = model.intercept_

# Calculate the regression line
regression_line = model.predict(x)
# Step 4: Calculate Mean Squared Error
mse = mean_squared_error(y, regression_line)

# Step 5: Print data, regression line, and MSE
print("Generated Data Points:")
for xi, yi in zip(x.flatten(), y):
    print(f"x: {xi:.2f}, y: {yi:.2f}")

print("\nRegression Line:")
print(f"y = {m:.2f}x + {b:.2f}")
print(f"\nMean Squared Error: {mse:.2f}")

# Plotting the data and regression line
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, regression_line, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Example')
plt.legend()
plt.show()
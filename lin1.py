import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.concatenate((np.zeros(1000000), np.array([1, 2, 3, 4, 5 ])))
y = np.concatenate((np.zeros(1000000), np.array([10, 11, 12, 13, 14 ])))

n = len(x)


# --- No-Intercept Regression ---
m = np.linalg.lstsq(x[:, np.newaxis], y, rcond=None)[0][0]
y_predicted_no_intercept = m * x

SST_no_intercept = np.sum(y**2)
SSE_no_intercept = np.sum((y - y_predicted_no_intercept)**2)
SSR_no_intercept = np.sum(y_predicted_no_intercept**2)
R_squared_no_intercept = 1 - (SSE_no_intercept / SST_no_intercept)

dfR_no_intercept = 1
dfE_no_intercept = n - 1
MSR_no_intercept = SSR_no_intercept / dfR_no_intercept
MSE_no_intercept = SSE_no_intercept / dfE_no_intercept
F_no_intercept = MSR_no_intercept / MSE_no_intercept
p_value_no_intercept = f.sf(F_no_intercept, dfR_no_intercept, dfE_no_intercept)

print("--- No-Intercept Regression ---")
print(f"Slope (m): {m:.4f}")
print(f"SST: {SST_no_intercept:.4f}")
print(f"SSR: {SSR_no_intercept:.4f}")
print(f"SSE: {SSE_no_intercept:.4f}")
print(f"F-statistic: {F_no_intercept:.4f}")
print(f"P-value: {p_value_no_intercept:.4f}")
print(f"R-squared: {R_squared_no_intercept:.4f}")

# --- With-Intercept Regression --- (for comparison)
X = np.vstack([np.ones(len(x)), x]).T
coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
b_intercept = coefficients[0]
m_intercept = coefficients[1]
y_predicted_with_intercept = m_intercept * x + b_intercept

y_mean = np.mean(y)
SST_with_intercept = np.sum((y - y_mean)**2)
SSE_with_intercept = np.sum((y - y_predicted_with_intercept)**2)
SSR_with_intercept = np.sum((y_predicted_with_intercept - y_mean)**2)
R_squared_with_intercept = 1 - (SSE_with_intercept / SST_with_intercept)

dfR_with_intercept = 1
dfE_with_intercept = n - 2
MSR_with_intercept = SSR_with_intercept / dfR_with_intercept
MSE_with_intercept = SSE_with_intercept / dfE_with_intercept
F_with_intercept = MSR_with_intercept / MSE_with_intercept
p_value_with_intercept = f.sf(F_with_intercept, dfR_with_intercept, dfE_with_intercept)

print("\n--- With-Intercept Regression ---")
print(f"Slope (m): {m:.4f}")
print(f"Intercept (b): {b_intercept:.4f}")
print(f"SST: {SST_with_intercept:.4f}")
print(f"SSR: {SSR_with_intercept:.4f}")
print(f"SSE: {SSE_with_intercept:.4f}")
print(f"F-statistic: {F_with_intercept:.4f}")
print(f"P-value: {p_value_with_intercept:.4f}")
print(f"R-squared: {R_squared_with_intercept:.4f}")

# --- Plotting ---
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='Data', color='blue')

# Calculate extended x-values for the lines
x_min = min(x.min(), 0) # Always include the origin
x_max = max(x) * 1.1  # Extend slightly beyond the data
x_extended = np.array([x_min, x_max])

# Calculate y-values for the extended lines
y_predicted_no_intercept_extended = m * x_extended
y_predicted_with_intercept_extended = m_intercept * x_extended + b_intercept
y_zero_extended = [0]* len(x_extended)

# Plot the extended lines
plt.plot(x_extended, y_predicted_no_intercept_extended, label='No Intercept', color='red', linestyle='--')
plt.plot(x_extended, y_predicted_with_intercept_extended, label='With Intercept', color='green')
plt.plot(x_extended, y_zero_extended, label='Zero Prediction', color='black', linestyle=':')

# Set plot limits to always include the origin
plt.xlim(x_min, x_max)
plt.ylim(min(0, min(y) * 1.1), max(y) * 1.1)  # Adjust y-axis limits dynamically

plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5)  # Add x-axis line
plt.axvline(0, color='black', linewidth=0.5)  # Add y-axis line
plt.title('Linear Regression: Guaranteed Negative R-squared (No-Intercept)')
plt.legend()
plt.grid(True)
plt.show()
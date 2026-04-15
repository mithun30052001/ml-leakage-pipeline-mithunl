import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Task 1: Create Synthetic Data
np.random.seed(42)

n = 60  #atleast 50 records is required

#features
area_sqft = np.random.randint(500, 3000, n)
num_bedrooms = np.random.randint(1, 5, n)
age_years = np.random.randint(0, 30, n)

# Creating price with some realistic relationship + noise
price_lakhs = (
    0.05 * area_sqft +
    5 * num_bedrooms -
    0.3 * age_years +
    np.random.normal(0, 10, n)  # noise
)

df = pd.DataFrame({
    "area_sqft": area_sqft,
    "num_bedrooms": num_bedrooms,
    "age_years": age_years,
    "price_lakhs": price_lakhs
})

# Features and Label
X = df[["area_sqft", "num_bedrooms", "age_years"]]
y = df["price_lakhs"]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Itercept and Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# Show first 5 actual vs predicted
results = pd.DataFrame({
    "Actual": y[:5],
    "Predicted": y_pred[:5]
})
print("\nFirst 5 Actual vs Predicted:\n", results)


# Task 2: Evaluation Metrics
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

# Explanation:
# MAE tells the average absolute error in predictions (lower is better).
# RMSE penalizes larger errors more heavily than MAE.
# R² shows how much variance in the target is explained by the model (closer to 1 is better).


# Task 3: Residual Analysis
residuals = y - y_pred

plt.figure()
plt.hist(residuals, bins=15)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Explanation:
# A residual is the difference between actual and predicted values.
# A roughly symmetric, bell-shaped histogram suggests the model errors are normally distributed,
# which indicates a good fit. Skewness or outliers may indicate model issues.
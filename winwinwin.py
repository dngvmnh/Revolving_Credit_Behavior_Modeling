import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Data: withdrawals and durations
withdrawals = [1099, 972, 1130, 1305, 953, 953, 1316, 1153, 906, 1109, 907, 907]
durations = [1, 2, 2, 1, 3, 0, 1, 1, 2, 5, 1, 2]

# Create a DataFrame
df = pd.DataFrame({'Withdrawals': withdrawals, 'Durations': durations})

# Split the data into features and target variable
X = df[['Durations']]
y = df['Withdrawals']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DecisionTreeRegressor with Hyperparameter Tuning
dt_model = DecisionTreeRegressor(random_state=42)
param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best DecisionTreeRegressor model
best_dt_model = grid_search.best_estimator_

# Predict and evaluate DecisionTreeRegressor
dt_y_pred = best_dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_y_pred)
print(f"DecisionTreeRegressor Mean Squared Error: {dt_mse:.2f}")

# RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=5)
rf_model.fit(X_train, y_train)

# Predict and evaluate RandomForestRegressor
rf_y_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)
print(f"RandomForestRegressor Mean Squared Error: {rf_mse:.2f}")

# Plot actual vs predicted values for DecisionTreeRegressor
plt.figure(figsize=(10, 6))
plt.scatter(y_test, dt_y_pred, color='blue', label='DecisionTree Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual Withdrawals')
plt.ylabel('Predicted Withdrawals')
plt.title('DecisionTreeRegressor: Actual vs Predicted Withdrawals')
plt.grid(True)
plt.legend()
plt.show()

# Plot actual vs predicted values for RandomForestRegressor
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_y_pred, color='green', label='RandomForest Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual Withdrawals')
plt.ylabel('Predicted Withdrawals')
plt.title('RandomForestRegressor: Actual vs Predicted Withdrawals')
plt.grid(True)
plt.legend()
plt.show()

# Print the test set with predictions for DecisionTreeRegressor
dt_results = X_test.copy()
dt_results['Actual Withdrawals'] = y_test
dt_results['Predicted Withdrawals'] = dt_y_pred
print("\nDecisionTreeRegressor: Test set with actual and predicted withdrawals:")
print(dt_results)

# Print the test set with predictions for RandomForestRegressor
rf_results = X_test.copy()
rf_results['Actual Withdrawals'] = y_test
rf_results['Predicted Withdrawals'] = rf_y_pred
print("\nRandomForestRegressor: Test set with actual and predicted withdrawals:")
print(rf_results)

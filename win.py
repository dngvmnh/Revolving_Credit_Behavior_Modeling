import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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

# Initialize and train the DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict the withdrawals on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual Withdrawals')
plt.ylabel('Predicted Withdrawals')
plt.title('Actual vs Predicted Withdrawals')
plt.grid(True)
plt.show()

# Print the test set with predictions
results = X_test.copy()
results['Actual Withdrawals'] = y_test
results['Predicted Withdrawals'] = y_pred
print("\nTest set with actual and predicted withdrawals:")
print(results)

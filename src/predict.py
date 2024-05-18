import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import csv

annual_interest_rate = 0.029  # Lãi suất cố định hàng năm
monthly_interest_rate = (1 + annual_interest_rate)**(1/12) - 1  # Lãi suất cố định mỗi tháng
months = 120  # Số tháng từ tháng 1/2025 đến tháng 12/2034

withdrawals = []
durations = []
interests = []
current_annual_interest_rates = []

file_path = 'data.csv'

with open(file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        so_tien_rut = int(row['So tien rut'])
        withdrawals.append(so_tien_rut)

with open(file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        thoi_gian_dao_han = int(row['Thoi gian dao han'])
        durations.append(thoi_gian_dao_han)

with open(file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        lai_thang = float(row['Lai thang'])
        interests.append(lai_thang)

with open(file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        lai_nam = float(row['Lai nam'])
        current_annual_interest_rates.append(lai_nam)
# Tạo DataFrame
df = pd.DataFrame({'Rút tiền': withdrawals, 'Thời gian đáo hạn': durations})

# Chia dữ liệu thành các biến đặc trưng và biến mục tiêu
X = df[['Thời gian đáo hạn']]
y = df['Rút tiền']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DecisionTreeRegressor với tinh chỉnh siêu tham số
dt_model = DecisionTreeRegressor(random_state=42)
param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Lấy mô hình DecisionTreeRegressor tốt nhất
best_dt_model = grid_search.best_estimator_

# Huấn luyện mô hình RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=5)
rf_model.fit(X_train, y_train)

# Dự đoán cho tương lai
months_future = 120  # Dự đoán cho 10 năm tiếp theo (từ 2035 đến 2044)

# Tạo DataFrame cho dự đoán
df_future = pd.DataFrame({'Thời gian đáo hạn': durations})

# Dự đoán với mô hình DecisionTreeRegressor
dt_future_pred = best_dt_model.predict(df_future)

# Dự đoán với mô hình RandomForestRegressor
rf_future_pred = rf_model.predict(df_future)

# Vẽ biểu đồ số tiền rút thực tế và dự đoán cho DecisionTreeRegressor
plt.figure(figsize=(14, 7))
plt.plot(range(months), withdrawals, color='blue', label='Rút tiền thực tế')
plt.plot(range(months, months + months_future), dt_future_pred, color='orange', label='Dự đoán DecisionTree')
plt.xlabel('Tháng')
plt.ylabel('Rút tiền (tỉ đồng)')
plt.title('DecisionTreeRegressor: Rút tiền thực tế và dự đoán trong tương lai')
plt.grid(True)
plt.legend()
plt.show()

# Vẽ biểu đồ số tiền rút thực tế và dự đoán cho RandomForestRegressor
plt.figure(figsize=(14, 7))
plt.plot(range(months), withdrawals, color='blue', label='Rút tiền thực tế')
plt.plot(range(months, months + months_future), rf_future_pred, color='green', label='Dự đoán RandomForest')
plt.xlabel('Tháng')
plt.ylabel('Rút tiền (tỉ đồng)')
plt.title('RandomForestRegressor: Rút tiền thực tế và dự đoán trong tương lai')
plt.grid(True)
plt.legend()
plt.show()

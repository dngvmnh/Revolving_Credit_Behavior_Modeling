import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Thiết lập các thông số của mô hình
mu = 800  # Kỳ vọng số tiền rút mỗi lần (tỉ đồng)
sigma = 200  # Độ lệch chuẩn số tiền rút mỗi lần (tỉ đồng)
lambda_rate = 1 / 3  # Tỷ lệ của phân phối mũ (kỳ vọng thời gian đáo hạn là 3 tháng)
lãi_suất = 0.029  # Lãi suất cố định mỗi tháng
months = 120  # Số tháng từ tháng 1/2025 đến tháng 12/2034

# Mô phỏng số tiền rút mỗi lần và thời gian đáo hạn
np.random.seed(42)  # Để đảm bảo tính tái lập
rút_tiền = np.random.normal(mu, sigma, months).astype(int)
durations = np.random.exponential(scale=1/lambda_rate + 1, size=months).astype(int)

# Giới hạn số tiền rút không quá 1000 tỉ đồng
rút_tiền = np.clip(rút_tiền, 0, 1000)

# Tạo DataFrame
df = pd.DataFrame({'Rút tiền': rút_tiền, 'Thời gian đáo hạn': durations})

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

# Dự đoán và đánh giá DecisionTreeRegressor
dt_y_pred = best_dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_y_pred)
print(f"Lỗi bình phương trung bình DecisionTreeRegressor: {dt_mse:.2f}")

# RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=5)
rf_model.fit(X_train, y_train)

# Dự đoán và đánh giá RandomForestRegressor
rf_y_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)
print(f"Lỗi bình phương trung bình RandomForestRegressor: {rf_mse:.2f}")

# Vẽ biểu đồ giá trị thực tế và dự đoán cho DecisionTreeRegressor
plt.figure(figsize=(10, 6))
plt.scatter(y_test, dt_y_pred, color='blue', label='Dự đoán DecisionTree')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Rút tiền thực tế')
plt.ylabel('Rút tiền dự đoán')
plt.title('DecisionTreeRegressor: Thực tế vs Dự đoán Rút tiền')
plt.grid(True)
plt.legend()
plt.show()

# Vẽ biểu đồ giá trị thực tế và dự đoán cho RandomForestRegressor
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_y_pred, color='green', label='Dự đoán RandomForest')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Rút tiền thực tế')
plt.ylabel('Rút tiền dự đoán')
plt.title('RandomForestRegressor: Thực tế vs Dự đoán Rút tiền')
plt.grid(True)
plt.legend()
plt.show()

# In tập kiểm tra với dự đoán cho DecisionTreeRegressor
dt_results = X_test.copy()
dt_results['Rút tiền thực tế'] = y_test
dt_results['Rút tiền dự đoán'] = dt_y_pred
print("\nDecisionTreeRegressor: Tập kiểm tra với rút tiền thực tế và dự đoán:")
print(dt_results)

# In tập kiểm tra với dự đoán cho RandomForestRegressor
rf_results = X_test.copy()
rf_results['Rút tiền thực tế'] = y_test
rf_results['Rút tiền dự đoán'] = rf_y_pred
print("\nRandomForestRegressor: Tập kiểm tra với rút tiền thực tế và dự đoán:")
print(rf_results)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Thiết lập các thông số của mô hình
mu = 100  # Kỳ vọng số tiền rút mỗi lần (tỉ đồng)
sigma = 10  # Độ lệch chuẩn số tiền rút mỗi lần (tỉ đồng)
lambda_rate = 1 / 5  # Tỷ lệ của phân phối mũ (kỳ vọng thời gian đáo hạn là 3 tháng)
lãi_suất = 0.029  # Lãi suất cố định mỗi tháng
months = 120  # Số tháng từ tháng 1/2025 đến tháng 12/2034

# Mô phỏng số tiền rút mỗi lần và thời gian đáo hạn
np.random.seed(42)  # Để đảm bảo tính tái lập
rút_tiền = np.random.normal(mu, sigma, months).astype(int)
durations = np.random.exponential(scale=1/lambda_rate , size=months).astype(int)+1

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

# Huấn luyện mô hình RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=5)
rf_model.fit(X_train, y_train)

# Dự đoán cho tương lai
months_future = 120  # Dự đoán cho 10 năm tiếp theo (từ 2035 đến 2044)
durations_future = np.random.exponential(scale=1/lambda_rate + 1, size=months_future).astype(int)

# Tạo DataFrame cho dự đoán
df_future = pd.DataFrame({'Thời gian đáo hạn': durations_future})

# Dự đoán với mô hình DecisionTreeRegressor
dt_future_pred = best_dt_model.predict(df_future)

# Dự đoán với mô hình RandomForestRegressor
rf_future_pred = rf_model.predict(df_future)

# Vẽ biểu đồ số tiền rút thực tế và dự đoán cho DecisionTreeRegressor
plt.figure(figsize=(14, 7))
plt.plot(range(months), rút_tiền, color='blue', label='Rút tiền thực tế')
plt.plot(range(months, months + months_future), dt_future_pred, color='orange', label='Dự đoán DecisionTree')
plt.xlabel('Tháng')
plt.ylabel('Rút tiền (tỉ đồng)')
plt.title('DecisionTreeRegressor: Rút tiền thực tế và dự đoán trong tương lai')
plt.grid(True)
plt.legend()
plt.show()

# Vẽ biểu đồ số tiền rút thực tế và dự đoán cho RandomForestRegressor
plt.figure(figsize=(14, 7))
plt.plot(range(months), rút_tiền, color='blue', label='Rút tiền thực tế')
plt.plot(range(months, months + months_future), rf_future_pred, color='green', label='Dự đoán RandomForest')
plt.xlabel('Tháng')
plt.ylabel('Rút tiền (tỉ đồng)')
plt.title('RandomForestRegressor: Rút tiền thực tế và dự đoán trong tương lai')
plt.grid(True)
plt.legend()
plt.show()

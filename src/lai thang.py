import numpy as np
import pandas as pd
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

# Tạo DataFrame từ dữ liệu mô phỏng
df = pd.DataFrame({
    'Tháng': pd.date_range(start='2014-01-01', periods=months, freq='M'),
    'Số tiền rút (tỉ đồng)': withdrawals,
    'Thời gian đáo hạn (tháng)': durations
})


monthly_update = []

for month in range(months):
    # Điều chỉnh lãi suất hàng năm với dao động nhỏ
    current_monthly_interest_rate = interests[month]
    current_annual_interest_rate = current_annual_interest_rates[month]
    
    # Lưu trữ thông tin cập nhật mỗi tháng
    monthly_update.append({
        'Tháng': df['Tháng'][month],
        'Số tiền rút (tỉ đồng)': withdrawals[month],
        'Lãi suất hàng tháng': current_monthly_interest_rate,
        'Lãi suất hàng năm': current_annual_interest_rate
    })

# Tạo DataFrame từ thông tin cập nhật mỗi tháng
df_update = pd.DataFrame(monthly_update)

# Xuất bảng cập nhật các tháng
print("\nBảng cập nhật các tháng:")
print(df_update)

# Vẽ biểu đồ thể hiện lãit theo tháng
fig, ax = plt.subplots(figsize=(14, 7))

ax.bar(df_update['Tháng'], df_update['Lãi suất hàng năm'], color='skyblue', width=10, )
ax.set_xlabel('Tháng')
ax.set_ylabel('Lãi theo năm')
ax.set_title('Lãi theo năm')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



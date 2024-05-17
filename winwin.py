import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Thiết lập các thông số của mô hình
mu = 1000  # Kỳ vọng số tiền rút mỗi lần (triệu đồng)
sigma = 200  # Độ lệch chuẩn số tiền rút mỗi lần (triệu đồng)
lambda_rate = 1 / 3  # Tỷ lệ của phân phối mũ (kỳ vọng thời gian đáo hạn là 3 tháng)
interest_rate = 0.029  # Lãi suất cố định mỗi tháng
months = 12  # Số tháng mô phỏng

# Mô phỏng số tiền rút mỗi lần và thời gian đáo hạn
withdrawals= [1099, 972, 1130, 1305, 953, 953, 1316, 1153, 906, 1109, 907, 907]
durations= [1, 2, 2, 1, 3, 0, 1, 1, 2, 5, 1, 2]

# Tạo DataFrame từ dữ liệu mô phỏng
df = pd.DataFrame({'Số tiền rút (triệu đồng)': withdrawals, 'Thời gian đáo hạn (tháng)': durations})

# Xuất bảng mô phỏng số tiền rút mỗi lần và thời gian đáo hạn
print("Bảng Mô phỏng số tiền rút mỗi lần và thời gian đáo hạn:")
print(df)

# Tính toán tổng chi phí lãi suất
total_interest = 0
balance = 0

# Lưu trữ lịch sử số dư và thời gian đáo hạn
balance_history = []

for month in range(months):
    # Rút tiền vào đầu tháng
    balance += withdrawals[month]
    balance_history.append((withdrawals[month], durations[month]))

    # Tính lãi suất trên số dư hiện tại
    interest = balance * interest_rate
    total_interest += interest

    # Cập nhật và trả nợ theo thời gian đáo hạn
    new_balance_history = []
    for amount, duration in balance_history:
        if duration <= 1:
            balance -= amount  # Trả nợ nếu đáo hạn
        else:
            new_balance_history.append((amount, duration - 1))
    balance_history = new_balance_history

print(f'\nTổng chi phí lãi suất trong {months} tháng: {total_interest:.2f} triệu đồng')

# Vẽ biểu đồ số tiền rút mỗi tháng và thời gian đáo hạn
fig, ax1 = plt.subplots()

ax1.set_xlabel('Tháng')
ax1.set_ylabel('Số tiền rút (triệu đồng)', color='tab:blue')
ax1.plot(range(1, months + 1), withdrawals, marker='o', color='tab:blue', label='Số tiền rút')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Thời gian đáo hạn (tháng)', color='tab:red')
ax2.plot(range(1, months + 1), durations, marker='x', color='tab:red', label='Thời gian đáo hạn')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
plt.title('Số tiền rút mỗi tháng và thời gian đáo hạn của công ty A')
plt.grid(True)
plt.show()
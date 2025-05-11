import pandas as pd

data = pd.read_csv('./datasets/daxDay.csv')
data['Date'] = pd.to_datetime(data['Date'])

data['Next_Close'] = data['Close'].shift(-1)  # Giá Close của ngày tiếp theo
data['Label'] = data['Next_Close'] > data['Close']  # True nếu tăng, False nếu giảm
data['Label'] = data['Label'].astype(int)  # Chuyển thành 0 hoặc 1


# Loại bỏ các hàng không hợp lệ (do shift)
data = data.dropna()


data.to_csv('./datasets/attention/trendLabel_daxDay.csv', index=False)
print(data.head())


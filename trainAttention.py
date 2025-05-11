from spEnv import SpEnv



# Giả sử sp_env là một instance của SpEnv
all_data = SpEnv.get_all_data_for_attention()

# In số lượng sample
print(f"Số lượng sample: {len(all_data)}")

# In một sample đầu tiên
hour_data, day_data, week_data = all_data[0]
print("Hour Data:", hour_data)
print("Day Data:", day_data)
print("Week Data:", week_data)
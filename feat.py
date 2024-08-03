import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 创建示例时间序列数据
time_series = pd.read_excel('../docs/pc_hour_clk.xlsx')
# 计算移动平均（MA）特征，窗口大小为12
time_series['MA_12'] = time_series['click_count'].rolling(window=12).mean()

# 计算标准差特征，窗口大小为12
time_series['STD_12'] = time_series['click_count'].rolling(window=12).std()
time_series['diff'] = time_series['click_count'].diff()
feat = time_series.iloc[11:]

plt.plot(feat['click_count'].iloc[::20], label='click_count')
plt.plot(feat['MA_12'].iloc[::20],  label='MA_12')
# plt.plot(time_series['STD_12'], label='STD_12')
plt.legend()
plt.show()
# feat.to_excel('../docs/feat.xlsx')
# 显示结果
print(feat.head(15))
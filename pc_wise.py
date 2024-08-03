# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# 数据
hour = list(range(24))
pc_click_count = [
    183149, 123372, 89275, 68900, 55523, 48723,
    60838, 100330, 391833, 859549, 1021332, 908763,
    468270, 648139, 964668, 1043661, 1004289, 804757,
    506160, 477497, 505062, 487001, 417954, 304576
]
wise_click_count = [
    2232942, 1472373, 1032418, 819002, 838909, 1265329,
    1966467, 2569713, 3492483, 4348065, 4770139, 4622352,
    4449597, 4400157, 4558039, 4708682, 4635390, 4350492,
    3904524, 3904658, 4329959, 4594285, 4310495, 3550695
]

# 创建折线图
plt.figure(figsize=(12, 6))
plt.plot(hour, pc_click_count, marker='o', label='PC Click Count', 
color='blue')
plt.plot(hour, wise_click_count, marker='o', label='Wise Click Count', 
color='orange')

# 添加标题和标签
plt.title('Click Counts by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Click Count')
plt.xticks(hour)  # 设置x轴刻度为0到23
plt.legend()
plt.grid()

# 显示图像
plt.show()

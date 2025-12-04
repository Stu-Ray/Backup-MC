import re
import os
import csv
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# 设置参数
factor = 0.81
repeat_count = 50
original_data_file = "./data.csv"
generated_data_dile = "./data-80.csv"

# 读取原始数据
df = pd.read_csv(original_data_file, encoding="gbk")

# 计算需要选择的行数
num_rows_to_copy = int(len(df) * factor)

# 随机选择行
random_rows = df.sample(n=num_rows_to_copy, random_state=1)

# 复制选择的行到原始数据
df_with_copies = pd.concat([df, random_rows], ignore_index=True)

# 重复数据集
df_with_copies = pd.concat([df_with_copies] * repeat_count, ignore_index=True)

# 打乱顺序
df_with_copies = df_with_copies.sample(frac=1, random_state=1).reset_index(drop=True)

# 生成新的行ID和订单ID
df_with_copies['行ID'] = range(1, len(df_with_copies) + 1)
df_with_copies['订单ID'] = [f"CN-2016-{str(i).zfill(7)}" for i in range(1, len(df_with_copies) + 1)]


# 生成递增的订单日期
start_date = datetime(2016, 1, 1)
order_dates = [start_date]

for _ in range(1, len(df_with_copies)):
    # 前一个日期 + 0或1天
    next_date = order_dates[-1] + timedelta(days=random.randint(0, 1))
    order_dates.append(next_date)

# 将订单日期格式化为字符串
df_with_copies['订单日期'] = [date.strftime('%Y/%m/%d') for date in order_dates]

# 生成发货日期，确保发货日期 >= 订单日期
df_with_copies['发货日期'] = [
    (datetime.strptime(date, '%Y/%m/%d') + timedelta(days=random.randint(0, 30))).strftime('%Y/%m/%d')
    for date in df_with_copies['订单日期']
]

# 选择并排序所需列
# final_df = random_rows[['行ID', '订单ID', '订单日期', '发货日期']]

# 保存到新文件
df_with_copies.to_csv(generated_data_dile, index=False)


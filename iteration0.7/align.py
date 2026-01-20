import pandas as pd 

# 1 .读取数据
df_p = pd.read_csv("sensor_pressure_50hz.csv")
df_h = pd.read_csv("sensor_heartrate_1hz.csv")

# 2 . 必需按时间排序
df_p = df_p.sort_values(by='ts')
df_h = df_h.sort_values(by='ts')

# 3. 使用merge_asof进行时间对齐
df_aligned = pd.merge_asof(df_p,df_h,on='ts',direction='backward')
print(df_aligned.head(20))
print(len(df_aligned))
print(f"NAN数据，共有{df_aligned['hr'].isna().sum()}个")
# 4. 保存对齐后的数据
print("保存对齐后的数据到iteration0.7/sensor_aligned.csv")
df_aligned.to_csv("sensor_aligned.csv",index=False)
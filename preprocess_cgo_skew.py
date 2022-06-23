# -*- coding: utf-8 -*-
"""
Created on Sun May  8 14:52:51 2022

@author: Phoenix
"""
import pandas as pd
import numpy as np
import copy
from datetime import datetime
'''
导入含换手率的数据
'''
root_path = r"C:\\Users\\Phoenix\\Desktop\\term_project\\data\\"
data_path = root_path+"raw_data.csv"
df = pd.read_csv(data_path,error_bad_lines=False)
index = df["date"] != "date" 
df = df[index]
df.dropna(how = "any",inplace=True)
column_list = list(df.columns)
del df[column_list[0]]
df["turnover"] = df["turnover"].apply(lambda x:float(x)/100)
df["p_change"] = df["p_change"].apply(lambda x:float(x)/100)
df.sort_values("date",inplace=True)
df=df.reset_index(drop=True)
first_firm = df[df["stock_code"]==1]
stock_list = np.unique(df["stock_code"])


'''
计算参考价格(rp)
'''
rps=[]
x = first_firm
size = len(x)
q = copy.deepcopy(x)
for j in range(size-99):
    q = copy.deepcopy(x)
    qq = q[j:j+100]
    turnover = qq["turnover"].to_numpy()
    price = qq["close"].to_numpy()
    # price = price*turnover
    keep = 1 - turnover
    sum_a = 0
    sum_p = 0
    for i in range(100):
        keep[i] = 1
        weight = keep.prod()
        a = turnover[i]*weight
        p = a*price[i]
        sum_a = sum_a +a
        sum_p = sum_p +p
    rp = sum_p/sum_a
    rps.append(rp)
rps_df = pd.DataFrame(rps)
rps_df.index = q.index[99:]
stock_list = list(stock_list)
del stock_list[0]

def cal_rp(df,stock_code):
    print(stock_code)
    rps=[]
    x = df[df["stock_code"]==stock_code]
    size = len(x)
    q = copy.deepcopy(x)
    for j in range(size-99):
        q = copy.deepcopy(x)
        qq = q[j:j+100]
        turnover = qq["turnover"].to_numpy()
        price = qq["close"].to_numpy()
        # price = price*turnover
        keep = 1 - turnover
        sum_a = 0
        sum_p = 0
        for i in range(100):
            keep[i] = 1
            weight = keep.prod()
            a = turnover[i]*weight
            p = a*price[i]
            sum_a = sum_a +a
            sum_p = sum_p +p
        rp = sum_p/sum_a
        rps.append(rp)
    rps = pd.DataFrame(rps)
    rps.index = q.index[99:]
    return rps


begin_time = datetime.now()
print(begin_time)
for stock_code in stock_list:
    rps = cal_rp(df,stock_code)
    rps_df = rps_df.append(rps)
end_time = datetime.now()
print(end_time)
duration = end_time-begin_time
print(duration)

rps_df = rps_df.reset_index()
df = df.reset_index()
data = pd.merge(df,rps_df,how="left")
# cgo因子表示当前价格与参考价格之间的差距
data["cgo"] = (data["close"]-data[0])/data["close"]
data = data.drop([0,1,2])
del data["index"]
data=data.reset_index(drop=True)
sk = df.groupby("stock_code")["close"].rolling(20).skew()
sk=sk.reset_index()
sk=sk.rename(columns={"close":"skew"})
sk.index = sk["level_1"]
data["skew"] = sk["skew"]
data.to_csv(root_path+"processed_data.csv")

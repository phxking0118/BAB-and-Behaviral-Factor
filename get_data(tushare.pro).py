# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:17:06 2022

@author: Phoenix
"""

'''
从tushare.pro获取不含换手率但是时间跨度更长的数据用以计算beta，存储于data_no_turnover.csv
'''
import tushare as ts
root_path = r"C:\\Users\\Phoenix\\Desktop\\term_project\\"
pro = ts.pro_api()
df = pro.daily(ts_code='000001.SZ')
date = df["trade_date"]
date = list(date)
df = pro.daily(trade_date='20220512')
date = date[:1500]
del date[0]
for i in date:
    print(i)
    df_temp = pro.daily(trade_date=i)
    df = df.append(df_temp)
    
df.to_csv(root_path + "data\\data_no_turnover.csv")




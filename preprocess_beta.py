# -*- coding: utf-8 -*-
"""
Created on Fri May 13 20:04:23 2022

@author: Phoenix
"""

import numpy as np
import copy
import pandas as pd
from datetime import datetime
root_path = r"C:\\Users\\Phoenix\\Desktop\\term_project\\"
df = pd.read_csv(root_path + "data\\data_no_turnover")
df = df.dropna(how="any")
df["stock_code"] = df["ts_code"].apply(lambda x:x[:6])
df["pct_chg"] = df["pct_chg"]/100
'''计算市场收益率rm，计算逻辑为个股收盘价按交易量加权求和后在时间上做差分'''
def mkt_p(x):
    dff = copy.deepcopy(x)
    s = dff["close"]*dff["vol"]/(dff["vol"].sum())
    s=s.sum()
    return s
mkt_price_per_share = df.groupby("trade_date").apply(lambda x:mkt_p(x))
rm = mkt_price_per_share.pct_change()
# rm["mkt_rtn"].cumsum().plot()
rm = rm.reset_index()
rm = rm.rename(columns={0:"mkt_rtn"})
rm = rm.dropna(how="any")
df = pd.merge(df,rm,how="left")

df = df.sort_values("trade_date")
df = df.reset_index(drop=True)
'''
遵循Frazzini2014以及期中因子复现中对beta的计算方式
使用250天作为滚动窗口由log（rtn）滚动计算波动性，以750天作为滚动窗口由连续三天的
log_rtn之和滚动计算相关性
'''
first_firm = df[df["stock_code"]=="000001"]

stock_list = list(np.unique(df["stock_code"]))
del stock_list[0]
first_firm["mkt_rtn_log"] = np.log(first_firm["mkt_rtn"]+1)
first_firm["mkt_rtn_3day_log"] = first_firm["mkt_rtn_log"].rolling(3).sum()

first_firm["pct_chg_log"] = np.log(first_firm["pct_chg"]+1)
first_firm["pct_chg_3day_log"] = first_firm["pct_chg_log"].rolling(3).sum()

first_firm = first_firm.dropna(how="any")
cor = first_firm[["pct_chg_3day_log","mkt_rtn_3day_log"]].rolling(750).corr()
cor = cor.reset_index()
cor = cor.dropna(how = "any")
cor.index = cor["level_0"]
del cor["level_0"]
del cor["level_1"]
del cor["pct_chg_3day_log"]
cor = cor[abs(cor["mkt_rtn_3day_log"]-1)>0.1]
cor = cor["mkt_rtn_3day_log"]

sigma_i = first_firm["pct_chg_log"].rolling(250).std().dropna(how="any")
sigma_m = first_firm["mkt_rtn_log"].rolling(250).std().dropna(how="any")
beta_df = cor*sigma_i/sigma_m
first_firm["beta"] = beta_df
def beta_cal(df,stock_code):
    firm = df[df["stock_code"]==stock_code]
    print(stock_code)
    firm["mkt_rtn_log"] = np.log(firm["mkt_rtn"]+1)
    firm["mkt_rtn_3day_log"] = firm["mkt_rtn_log"].rolling(3).sum()

    firm["pct_chg_log"] = np.log(firm["pct_chg"]+1)
    firm["pct_chg_3day_log"] = firm["pct_chg_log"].rolling(3).sum()

    firm = firm.dropna(how="any")
    cor = firm[["pct_chg_3day_log","mkt_rtn_3day_log"]].rolling(750).corr()
    cor = cor.reset_index()
    cor = cor.dropna(how = "any")
    cor.index = cor["level_0"]
    del cor["level_0"]
    del cor["level_1"]
    del cor["pct_chg_3day_log"]
    cor = cor[abs(cor["mkt_rtn_3day_log"]-1)>0.1]
    cor = cor["mkt_rtn_3day_log"]

    sigma_i = firm["pct_chg_log"].rolling(250).std().dropna(how="any")
    sigma_m = firm["mkt_rtn_log"].rolling(250).std().dropna(how="any")
    beta = cor*sigma_i/sigma_m
    return beta
begin_time = datetime.now()
print(begin_time)
for stock_code in stock_list:
    beta = beta_cal(df,stock_code)
    beta_df = beta_df.append(beta)
end_time = datetime.now()
print(end_time)
duration = end_time-begin_time
print(duration)



beta_df = beta_df.dropna(how="any")

df["beta"] = beta_df
df = df.dropna(how="any")
'''打开存储cgo和skew因子的文件processed_data.csv,将计算好的beta加入processed_data中
   存储在final_data.csv中'''
data = pd.read_csv(root_path+"data\\processed_data.csv")
def standard(x):
    year = x[:4]
    month = x[4:6]
    day = x[6:8]
    date = year+"-"+month+"-"+day
    return date
df["trade_date"] = df["trade_date"].apply(lambda x:str(x))
df["trade_date"] = df["trade_date"].apply(lambda x:standard(x))

df[["beta","pct_chg"]].corr()
col_list = df.columns
del df[col_list[0]]
df = df.rename(columns = {"trade_date":"date"})
df_beta = df[["date","beta","stock_code"]]
df_beta["stock_code"] = df_beta["stock_code"].apply(lambda x:int(x))
del data["beta"]
data = pd.merge(data,df_beta,how="left")
data = data.dropna()
a = list(np.unique(data["date"]))
p = data["date"]!="28"
data=data[p]
data[["beta","p_change"]].corr()
data.to_csv(root_path+"data\\final_data.csv")


# -*- coding: utf-8 -*-
"""
Created on Sat May 14 10:26:33 2022

@author: Phoenix
"""
import pandas as pd
import copy
import numpy as np
root_path = r"C:\\Users\\Phoenix\\Desktop\\term_project"
# 导入数据
data = pd.read_csv(root_path+"\\data\\final_data.csv")
data = data.sort_values("date")
std_beta = data.groupby("date")["beta"].std()
std_beta = std_beta.reset_index(drop=True)
std_beta.plot(title="the std of beta",xlabel="date")
std_beta = data.groupby("date")["beta"].mean()
std_beta = std_beta.reset_index(drop=True)
std_beta.plot()

'''按照文章所给方式单独计算bab因子'''
def cal_bab(x):
    data_one = copy.deepcopy(x)
    beta_rank = data_one["beta"].rank()
    beta = data_one["beta"]/(data_one["beta"].mean())
    rtn = data_one["p_change"]
    n = len(beta_rank)
    average_rank = beta_rank.sum()/n
    wh = beta_rank.copy()
    wh[wh<=average_rank] = 0
    wh[wh>average_rank] = 2
    wh=wh*beta_rank
    abs_sum = abs(beta_rank-average_rank).sum()
    wh =wh/(3*abs_sum)
    wl = beta_rank.copy()
    wl[wl<=average_rank] = 2
    wl[wl>average_rank] = 0
    wl=wl*beta_rank
    abs_sum = abs(beta_rank-average_rank).sum()
    wl =wl/abs_sum
    beta_l = (beta*wl).sum()
    beta_h = (beta*wh).sum()
    rtn_l = (rtn*wl).sum()
    rtn_h = (rtn*wh).sum()
    bab = rtn_l/beta_l - rtn_h/beta_h
    return bab

bab = data.groupby("date").apply(lambda x:cal_bab(x))
bab.cumsum().plot()



data["p_chg"] = data.groupby("stock_code")["p_change"].shift(-1)
data = data.dropna(how="any")
'''根据数据进行因子组合构建'''
def cal_port(x,hlm,factor,Quantile):
    data_one = copy.deepcopy(x)
    rank = data_one[factor].rank()
    n = len(rank)
    d = int(n/Quantile)
    if hlm=="h":
        a = rank>n-d
    if hlm=="l":
        a = rank<d
    if hlm=="m":
        a = (rank<n-d)&(rank>d)
    return a

cgo_h = data.groupby("date").apply(lambda x:cal_port(x,hlm="h",factor="cgo",Quantile=3))
cgo_m = data.groupby("date").apply(lambda x:cal_port(x,hlm="m",factor="cgo",Quantile=3))
cgo_l = data.groupby("date").apply(lambda x:cal_port(x,hlm="l",factor="cgo",Quantile=3))

skew_h = data.groupby("date").apply(lambda x:cal_port(x,hlm="h",factor="skew",Quantile=3))
skew_m = data.groupby("date").apply(lambda x:cal_port(x,hlm="m",factor="skew",Quantile=3))
skew_l = data.groupby("date").apply(lambda x:cal_port(x,hlm="l",factor="skew",Quantile=3))

beta_h = data.groupby("date").apply(lambda x:cal_port(x,hlm="h",factor="beta",Quantile=2))
beta_l = data.groupby("date").apply(lambda x:cal_port(x,hlm="l",factor="beta",Quantile=2))

cgo_list = [cgo_h,cgo_m ,cgo_l]
skew_list = [skew_h,skew_m,skew_l]
beta_list = [beta_h ,beta_l]
portfolio_list = [cgo_h.copy()]*18


for i in range(3):
    for j in range(3):
        for k in range(2):
            a = cgo_list[i]&skew_list[i]&beta_list[k]
            portfolio_list[i+j+k] = a


construct_cgo_h = portfolio_list[0]
for i in range(6):
    construct_cgo_h = construct_cgo_h|portfolio_list[i]
    
construct_cgo_l = portfolio_list[12]
for i in range(12,18):
    construct_cgo_l = construct_cgo_l|portfolio_list[i]

construct_skew_h = portfolio_list[0]
for i in [0,1,6,7,12,13]:
    construct_skew_h = construct_skew_h|portfolio_list[i]

construct_skew_l = portfolio_list[4]
for i in [4,5,10,11,16,17]:
    construct_skew_l = construct_skew_l|portfolio_list[i]

construct_beta_h = portfolio_list[0]
for i in range(9):
    construct_beta_h = construct_beta_h|portfolio_list[2*i]

construct_beta_l = portfolio_list[1]
for i in range(9):
    construct_beta_l = construct_beta_l|portfolio_list[2*i-1]

construct = [construct_cgo_h,construct_cgo_l,construct_skew_h,construct_skew_l,
             construct_beta_h,construct_beta_l]

for i in range(6):
    construct[i] = construct[i].reset_index()
    construct[i].index = construct[i]["level_1"]
    del construct[i]["date"]
    del construct[i]["level_1"]
data["cgo_i"] = construct[0][0]
data["long_port"] = data["cgo_i"]*data["p_chg"]
'''计算组合平均回报以得到因子值'''
def portfolio_rtn(x,factor):
    data_one = copy.deepcopy(x)
    if factor=="cgo":
        short = construct[0][0]
        long = construct[1]["cgo"]
    if factor=="skew":
        short = construct[2][0]
        long = construct[3][0]
    if factor=="beta":
        short = construct[4][0]
        long = construct[5][0]
    data_one["long"] = long
    data_one["short"] = short
    data_one["long_p"] =  data_one["p_chg"]*long
    data_one["short_p"] = data_one["p_chg"]*short
    rtn = data_one.groupby("date")["long_p"].mean()-\
        data_one.groupby("date")["short_p"].mean()

    return rtn

cgo_rtn = portfolio_rtn(data,"cgo")
skew_rtn = portfolio_rtn(data,"skew")
beta_rtn = portfolio_rtn(data,"beta")
cgo_rtn = cgo_rtn.reset_index()
cgo_rtn = cgo_rtn.rename(columns = {0:"cgo_rtn"})
skew_rtn = skew_rtn.reset_index()
skew_rtn = skew_rtn.rename(columns = {0:"skew_rtn"})
beta_rtn = beta_rtn.reset_index()
beta_rtn = beta_rtn.rename(columns = {0:"beta_rtn"})
data = pd.merge(data,cgo_rtn,how="left")
data = pd.merge(data,skew_rtn,how="left")
data = pd.merge(data,beta_rtn,how="left")


cgo_rtn.cumsum().plot(title="V-shaped Factor Cumulative Return")
skew_rtn.cumsum().plot()
beta_rtn.cumsum().plot()
cgo_rtn["cgo_rtn"].corr(skew_rtn["skew_rtn"])
beta_rtn["beta_rtn"].corr(skew_rtn["skew_rtn"])



'''计算市场回报,为什么在算beta的时候算过而在这里要再算一次...
主要是因为当时只把beta值加入processed_data中了，忘了市场回报'''

df = pd.read_csv(root_path + "\\data\\data_no_turnover.csv")
df = df.dropna(how="any")
df["stock_code"] = df["ts_code"].apply(lambda x:x[:6])
df["pct_chg"] = df["pct_chg"]/100
def mkt_p(x):
    dff = copy.deepcopy(x)
    s = dff["close"]*dff["vol"]/(dff["vol"].sum())
    s=s.sum()
    return s
mkt_price_per_share = df.groupby("trade_date").apply(lambda x:mkt_p(x))
rm = mkt_price_per_share.pct_change()
rm = rm.reset_index()
rm = rm.rename(columns={0:"mkt_rtn"})
rm = rm.dropna(how="any")
rm = rm.rename(columns={"trade_date":"date"})

data = data.dropna()
def standard(x):
    year = x[:4]
    month = x[4:6]
    day = x[6:8]
    date = year+"-"+month+"-"+day
    return date
rm["date"] = rm["date"].apply(lambda x:str(x))
rm["date"] = rm["date"].apply(lambda x:standard(x))
data = pd.merge(data,rm,how="left")

'''定义回归函数'''
def linear_regression(X,y):
    XTX = np.dot(X.T,X)
    XTX_inv = np.linalg.inv(XTX)
    XTY = np.dot(X.T,y)
    beta = np.dot(XTX_inv,XTY)
    return beta
def timeseries_regresstion(x):
    df = copy.deepcopy(x)
    df['open'] = 1
    Y = df['p_change'].to_numpy()
    Y = np.reshape(Y,(-1,1))
    X = df[["open","cgo_rtn","skew_rtn","beta_rtn","mkt_rtn"]].to_numpy()
    #如果XTX是奇异矩阵，那么返回nan，反之进行回归
    if np.linalg.det(np.dot(X.T,X)) == 0:
        beta = np.nan
    else:
        beta = linear_regression(X, Y)
    return beta
'''定义计算仓位函数'''
def position_long(x):
    df = copy.deepcopy(x)
    pred_rank = df["coef_0"].rank()
    len_ = len(pred_rank)
    long = pred_rank > len_-500
    return long

def position_short(x):
    df = copy.deepcopy(x)
    pred_rank = df["coef_0"].rank()
    short = pred_rank < 500
    return short
def standard_reverse(x):
    year = x[:4]
    month = x[5:7]
    day = x[8:10]
    date = year+month+day
    return int(date)
def unique(x):
    df = copy.deepcopy(x)
    a = df["coef_0"].mean()
    return a
data["date"] = data["date"].apply(lambda x:standard_reverse(x))
period = list(np.unique(data["date"]))
n = len(period)

'''根据策略回测'''
i=0
p_data = data[(data["date"]>=period[i*20])&(data["date"]<period[i*20+20])]
coef = p_data.groupby("stock_code").apply(lambda x:timeseries_regresstion(x))
coef =coef.reset_index()
coef = coef.rename(columns={0:"coef"})
p_data = pd.merge(p_data,coef)
predict = p_data[["p_change","stock_code","date","coef","cgo_rtn","beta_rtn","skew_rtn","mkt_rtn"]]
predict["coef_0"] = predict["coef"].apply(lambda x: float(x[0]))
a = predict.groupby("stock_code").apply(lambda x:unique(x))
a = a.reset_index()
pred_rank = a[0].rank()
len_ = len(pred_rank)
long = pred_rank > len_-500
short = pred_rank < 500
a["long"] = long
a['short'] = short
a["date"] = period[20*i+19]
a_df = a.copy()
for i in range(1,20):
    print(i)
    p_data = data[(data["date"]>=period[i*20])&(data["date"]<period[i*20+20])]
    coef = p_data.groupby("stock_code").apply(lambda x:timeseries_regresstion(x))
    coef =coef.reset_index()
    coef = coef.rename(columns={0:"coef"})
    p_data = pd.merge(p_data,coef)
    predict = p_data[["p_change","stock_code","date","coef","cgo_rtn","beta_rtn","skew_rtn","mkt_rtn"]]
    predict["coef_0"] = predict["coef"].apply(lambda x: float(x[0]))
    a = predict.groupby("stock_code").apply(lambda x:unique(x))
    a = a.reset_index()
    pred_rank = a[0].rank()
    len_ = len(pred_rank)
    long = pred_rank > len_-500
    short = pred_rank < 500
    a["long"] = long
    a['short'] = short
    a["date"] = period[20*i+19]
    a_df = a_df.append(a)
    
del a_df[0]
q = pd.merge(data,a_df,how="left")
q = q.groupby("stock_code").fillna(method="ffill")
q = q.dropna()


'''计算策略回报'''
def portfolio_rtn(x):
    data_one = copy.deepcopy(x)
    data_one["long_p"] =  data_one["p_change"]*data_one["long"]
    data_one["short_p"] = data_one["p_change"]*data_one["short"]
    rtn = data_one.groupby("date")["long_p"].mean()-\
        data_one.groupby("date")["short_p"].mean()
    return rtn
rtn = portfolio_rtn(q)
rtn = rtn.reset_index(drop=True)
rtn.cumsum().plot(title="Cumulative yield of portfolio")


# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:50:28 2022

@author: Phoenix
"""

'''
从tushare.org获取含换手率的交易数据，存储于raw_data.csv
'''
from concurrent.futures import ThreadPoolExecutor
import tushare as ts
root_path = r"C:\Users\Phoenix\Desktop\term_project\data\\"
pro = ts.pro_api()
data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
stock_list = list(data["symbol"])
a = ts.get_hist_data("000001")
a['stock_code'] = str("000001")

def get_data(root_path,stock_code,include_index):
    print(stock_code+"th stock's data downloading!")
    a = ts.get_hist_data(stock_code)
    a['stock_code'] = stock_code
    a.to_csv(root_path+"raw_data.csv",index = False ,mode='a')
        
if __name__ == '__main__':
    with ThreadPoolExecutor(5) as t:
        for stock_code in stock_list:
            t.submit(get_data, root_path = root_path, stock_code = stock_code)



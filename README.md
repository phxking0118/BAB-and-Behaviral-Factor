# BAB-and-Behaviral-Factor
- 文件夹中程序为完成报告的辅助文件，使用Python3.7编写。
- get_data.py是从tushare获取数据的程序，分为从org和pro获取数据的程序，获取数据分别保存于raw_data.csv和data_without_turnover.csv。
- preprocessed是数据预处理的程序，其中cgo和skew由raw_data.csv计算所得，beta由data_without_turnover.csv计算所得。由于所用数据不同，所以分为两部分程序。
- portfolio_construct.py是策略构建的主程序。

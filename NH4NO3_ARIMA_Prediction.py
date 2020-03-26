import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning

#1 数据读取
io='C:/Users/23008/Desktop/AJPY/金泽1号测亭氨氮.xlsx'
data=pd.read_excel(io,header=None)
#print(data)

# 数据转换
dta= data[1]
dta=np.array(dta,dtype=np.float) 
dta=pd.Series(dta)
dta.index = pd.to_datetime(data[2])
print(dta.head())

# 缺省值
print(dta.isnull().sum().max())

# 时间重取样
da=dta.resample('D').mean() # 时间重取样后出现了空值

# 异常值处理
for i in range(len(da)):
	if da[i]>0.5 or da[i]<0.15:
		da[i]=da.mean()

# 缺省值填充
x=da.fillna(method="ffill")
print(x.head())
plt.plot(x)
plt.show()



#ts_log=np.log(x)
# moving_avg = ts_log.rolling(window=12).mean()
# plt.plot(ts_log ,color = 'blue')
# plt.plot(moving_avg, color='red')
# plt.show()

#ts_log_diff = ts_log - ts_log.shift()
# ts_log_diff.dropna(inplace=True)
# test_stationarity(ts_log_diff)


#切分数据集
test_point = 20  #测试点数
train_size = int(len(x) - test_point)  #测试集大小
train, test =x[12:train_size], x[train_size:]  #切分数据集

p,d,q = 1, 0, 1
model = ARIMA(endog=train, order=(p, d, q)).fit(disp=-1) # 自回归函数p,差分d,移动平均数q
predictions2 = model.forecast(test_point) # 连续预测N个值# disp<0:不输出过程
predictions_series2= pd.Series(predictions2[0], index=test.index)
y =predictions_series2

# y =np.exp(predictions_series2)



mse = ((x - y)**2).mean()
rmse = np.sqrt(mse)
mape = ((y-x).abs()/x).mean()

#mape=np.sqrt(sum((y-x)**2)/len(x))

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(20, 8),facecolor='w')
plt.plot(x, 'r-', lw=2, label='原始数据')
plt.plot(y, 'g-', lw=2, label='预测数据')
title = 'ARIMA预测(AR=%d, d=%d, MA=%d)：RMSE=%.4f,MAPEE=%.4f' % (p, d, q, rmse,mape)
plt.legend(loc='upper left')
plt.grid(b=True, ls=':')
plt.title(title, fontsize=16)
plt.tight_layout(2)
    # plt.savefig('%s.png' % title)
plt.show()
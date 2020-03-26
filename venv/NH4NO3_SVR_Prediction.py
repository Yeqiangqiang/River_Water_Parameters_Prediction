import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib as mpl
import matplotlib.pyplot as plt

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
dtaa=da.fillna(method="ffill")


X = np.mat(range(1, len(dtaa.data)  + 1)).T
y = dtaa.data



svr_rbf = SVR(kernel='rbf', C=1, gamma=0.1)
y_rbf = svr_rbf.fit(X, y).predict(X)

mse = ((y_rbf - y)**2).mean()
rmse = np.sqrt(mse)
# Result
_, ax = plt.subplots(figsize=[14, 7])
ax.scatter(np.array(X), np.array(y), c='#F61909', label='data', s=8)
ax.plot(X, y_rbf, c='#486D0B', label='RBF Model')
# ax.plot(X, y, c='#000000', label='CLOSE')
ticks = ax.get_xticks()
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression,RMSE=%.4f'% (rmse))
plt.legend()
plt.show()
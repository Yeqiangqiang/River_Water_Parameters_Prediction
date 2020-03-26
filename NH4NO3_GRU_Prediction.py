import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import time

# 1数据读取
io='C:/Users/23008/Desktop/AJPY/金泽1号测亭氨氮.xlsx'
data=pd.read_excel(io,header=None)
#print(data)

array= data[1]
# 构建数据集 (本例中，通过最近的 5 个值预测下一个值)
listX = [] 
listy = []
X = {} 
y = {}

for i in range(0,len(array) - 6):
    listX.append(array[i:i+5].values.reshape([5,1])) 
    listy.append(array[i+6])



arrayX = np.array(listX)    #  ((5255, 5, 1), (5255,))
arrayy = np.array(listy)    #  (140250, 1)

# 划分数据集
X_train=arrayX[0:4500]
X_test=arrayX[4500:5255]

y_train=arrayy[0:4500]
y_test=arrayy[4500:5255]




from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential



def build_model(layers):  #layers [1,50,100,1]
    model = Sequential()

    #Stack LSTM
    model.add(GRU(input_dim=layers[0],output_dim=layers[1],return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(layers[2],return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

model = build_model([1, 5, 100, 1])


# Fit the model to the data 
model.fit(X_train,y_train,batch_size=512,epochs=10,validation_split=0.08)
test_results = model.predict(X_test)



# Rescale the test dataset and predicted data 
test_results = test_results * 7.172300e+06 + 5.761423e+06 
y_test = y_test * 7.172300e+06 + 5.761423e+06 

plt.subplot() 
plot_predicted, = plt.plot(test_results,label='predicted')

plot_test, = plt.plot(y_test,label='test')
plt.legend(handles= [plot_predicted,plot_test])
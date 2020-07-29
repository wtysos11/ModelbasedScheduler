# 尝试建模
# 输入单机流量和流量
# 输出CPU占用率

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib import pyplot as plt

import os

# dirName = ['qlearning-1','qthreshold-1','sarsa-arima-1','qlearning-2','qthreshold-2','sarsa-arima-2']
dirName = ['0_workload1','2_workload2','4_workload1','6_workload2','8_workload1','10_workload2','12_workload1','14_workload2','0_2_workload2','2_2_workload2','4_2_workload2','23_2_workload2']

cpuFilename = 'cpuUtilization.log'
responseFilename = 'responseTime.log'
throughputFilename = 'requestRate.log'
supplypodFilename = 'podSupply.log'

anscpuList = []
ansresponseList = []
ansthroughputList = []
anssupplypodList = []

for dirname in dirName:
    cpuReader = open(dirname+'/'+cpuFilename,'r')
    responseReader = open(dirname+'/'+responseFilename,'r')
    throughputReader = open(dirname+'/'+throughputFilename,'r')
    supplypodReader = open(dirname+'/'+supplypodFilename,'r')
    
    cpuList = cpuReader.readlines()
    responseList = responseReader.readlines()
    throughputList = throughputReader.readlines()
    supplypodList = supplypodReader.readlines()
    
    cpuReader.close()
    responseReader.close()
    throughputReader.close()
    supplypodReader.close()
    
    #将行尾带有\n的文本转成float
    def floatTranslator(element):
        return float(element[:-1])
    
    #将行尾带有\n的文本转成int
    def intTranslator(element):
        return int(element[:-1])
    
    anscpuList.append(list(map(floatTranslator,cpuList)))
    ansresponseList.append(list(map(floatTranslator,responseList)))
    ansthroughputList.append(list(map(floatTranslator,throughputList)))
    anssupplypodList.append(list(map(intTranslator,supplypodList)))

#拆分成监督数据
cpuList = []
resList = []
podList = []
trafficList = []

totalX = []

for botton in range(len(anscpuList)):
    for index in range(len(anscpuList[botton])):
        cpuList.append(anscpuList[botton][index])
        resList.append(ansresponseList[botton][index])
        podList.append(anssupplypodList[botton][index])
        avgTraffic = 0
        if anssupplypodList[botton][index]>0:
            avgTraffic = ansthroughputList[botton][index]/anssupplypodList[botton][index]
            trafficList.append(ansthroughputList[botton][index]/anssupplypodList[botton][index])
        else:
            trafficList.append(0)
        totalX.append([avgTraffic,anssupplypodList[botton][index]])

#标准训练过程
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
def getModel():
    '''
    组建预测模型，进行拼装
    '''
    #model = SVR(kernel='rbf',gamma='scale') #SVR模型，CPUMSE130左右/5，responseTIme 1W1/45
    #model = RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, 
    #          scoring=None, cv=5, gcv_mode=None, store_cv_values=False)# CPU MSE 96/6, responseTime 1W4/62 ,结果为一条直线
    #model = KNeighborsRegressor(n_neighbors=5, weights="uniform", algorithm="auto", 
    #                      leaf_size=30, p=2, metric="minkowski", metric_params=None)#CPU 98/6,respontime 9k/45,结果不错
    #model = SVR(kernel="rbf", degree=3, gamma="auto", coef0=0.0, 
    #      tol=0.001, C=1.0, epsilon=0.1, shrinking=True, 
    #      cache_size=200, verbose=False, max_iter=-1) #200/7, 2w2/63
    #model = BaggingRegressor(n_estimators=100)#90/6,1w/47
    #model = RandomForestRegressor(n_estimators=500,criterion='mse',oob_score=True)
    model = GradientBoostingRegressor(n_estimators=500,loss='huber')
    #model = AdaBoostRegressor()
    return model

# RandomForest mse 89/6,1w/47
# GBDT ls 71/5,8k9/43
# GBDT lad 70/5,8k4/36
# GBDT huber 72/5,7k5,36
# GBDT quantile 198/9,1w5/79
# Adaboost 90/6,1w8/109
from joblib import dump, load
def trainAndTest(X,y,ratio = 0.3):
    model = getModel()
    data_train, data_test, label_train, label_test = train_test_split(X, y, test_size = ratio)
    model.fit(data_train,label_train)
    result = model.predict(data_test)
    return mean_squared_error(result,label_test),np.average(np.abs(result-label_test)),model
    #绘制散点图，X轴为平均流量，Y轴为预测结果
    #for i in range(len(result)):
    #    plt.scatter(data_test[i][0],result[i],color='red',alpha=0.6)
    #    plt.scatter(data_test[i][0],label_test[i],color='blue',alpha=0.6)
    #plt.show()
    #分析哪个频段的预测效果最差
    #div = np.zeros(3)
    #count = np.zeros(3)
    #for i in range(len(result)):
    #    if label_test[i] < 50:
    #        div[0] += np.abs(result[i]-label_test[i])
    #        count[0] += 1
    #    elif label_test[i] < 250:
    #        div[1] += np.abs(result[i]-label_test[i])
    #        count[1] += 1
    #    else:
    #        div[2] += np.abs(result[i]-label_test[i])
    #        count[2] += 1
    #for k in range(3):
    #    print(div[k]/count[k])

#训练关于CPU的模型
n1x = []
n1y = []
n2x = []
n2y = []
m1 = []
m2 = []
for i in range(10):
    t1,t2,ma = trainAndTest(totalX,cpuList)
    k1,k2,mb = trainAndTest(totalX,resList)
    n1x.append(t1)
    n1y.append(t2)
    n2x.append(k1)
    n2y.append(k2)
    m1.append(ma)
    m2.append(mb)

print('CPU')
print(np.average(t1))
print(np.average(t2))
print('responseTime')
print(np.average(k1))
print(np.average(k2))

n1x = np.array(n1x)
n2x = np.array(n2x)
index1 = np.where(n1x==min(n1x))[0][0]
index2 = np.where(n2x==min(n2x))[0][0]
dump(m1[index1],'CPUModel.m')
dump(m2[index2],'responseTimeModel.m')
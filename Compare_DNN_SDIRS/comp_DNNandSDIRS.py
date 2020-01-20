import numpy as np
import math as math
import matplotlib.pyplot as plt
from SDAlgo_learntR import *
from SDAlgo_SDIRS import *
import csv
pltx = []
rangee = range(10,22,2)
plty = [[0 for i in rangee] for _ in range(2)]

m = 10
n = 10


x = np.zeros((m,1))
H = np.zeros((n,m))
Radius = [0 for _ in range(3)]
# print(H)
# print(x)
# print(len(data_prediction))

for dB in rangee:
    pltx.append(dB)
    with open('../DNN/TestData'+ str(dB) + 'dB.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    with open('../DNN/TestDataPrediction'+str(dB) + 'dB.csv', newline='') as csvfile:
        data_prediction = list(csv.reader(csvfile))

    SNRdB = dB
    SNR  =  math.pow(10,SNRdB/10)
    variance = (15.0 * m) / (SNR * 6)
    # print(data_prediction[0])
    cnt = 0
    print(dB)
    for _i in range(0,2000) :
        
        ###reading the data
        for _j in range(0,m):
            x[_j][0] = float(data[_i][_j]) 
        cur = 10
        for _j in range(0,n):
            for _k in range(0,m):
                H[_j][_k] = float(data[_i][cur])
                cur += 1
        for _j in range(0,3):
            Radius[_j] = float(data_prediction[_i][_j])
        

        flopsCount,ans,answer = SD_learntR(m,n,H,x,4,Radius)
        plty[0][cnt] += (math.log(flopsCount,m))/2000

        flopsCount,ans,answer = SD_SDIRS(m,n,H,x,variance,QAM = 4)
        plty[1][cnt] += (math.log(flopsCount,m))/2000
    
    cnt += 1
    
plt.plot(pltx,plty[0],'go-',label= 'DNN')
plt.plot(pltx,plty[1],'ys-',label= 'SDIRS')
plt.ylabel('Log10(Flop(Avg))')
plt.xlabel('dB')
plt.title("N = M = 10, through 10dB - 20dB")
plt.legend()
plt.show()



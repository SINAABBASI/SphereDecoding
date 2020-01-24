import numpy as np
import math as math
import matplotlib.pyplot as plt
from SDAlgo_learntR import *
from SDAlgo_SDIRS import *
import time
import csv
pltx = []
rangee = range(8,22,2)
plty = [[0 for i in rangee] for _ in range(2)]
plty_NL = [[0 for i in rangee] for _ in range(2)]
m = 10
n = 10


x = np.zeros((m,1))
H = np.zeros((n,m))
Radius = [0 for _ in range(3)]

cnt = 0
for dB in rangee:

    pltx.append(dB)
    with open('../DNN/TestData'+ str(dB) + 'dB.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    with open('../DNN/TestDataPrediction'+str(dB) + 'dB.csv', newline='') as csvfile:
        data_prediction = list(csv.reader(csvfile))

    SNRdB = dB
    SNR  =  math.pow(10,SNRdB/10)
    print(SNR)
    variance = (m*15) / (SNR * 6)
    # print(data_prediction[0])
    
    print(dB)
    for _i in range(0,2000) :
        
        ###reading the data
        for _j in range(0,m):
            x[_j][0] = float(data[_i][_j]) 
        cur = m
        for _j in range(0,n):
            for _k in range(0,m):
                H[_j][_k] = float(data[_i][cur])
                cur += 1
        for _j in range(0,3):
            Radius[_j] = float(data_prediction[_i][_j])
        
        # print(H)  
        start = time.time()
        flopsCount,ans,N_lattice = SD_learntR(m,n,H,x,4,Radius)
        plty[0][cnt] += (time.time() - start)/2000
        plty_NL[0][cnt] += math.log10(N_lattice) / 2000
        # print(Radius[i])
        start = time.time()
        flopsCount,ans,N_lattice = SD_SDIRS(m,n,H,x,variance, 4)
        plty[1][cnt] += ((time.time() - start))/2000
        plty_NL[1][cnt] += math.log10(N_lattice) / 2000
        # print(N_lattice)
        # print("next")
    cnt += 1

for i in range(len(plty[0])):
    plty[0][i] /= plty[1][i] 

plt.yscale('log')
plt.plot(pltx,plty[0],'ro-')
# plt.plot(pltx,plty[1],'bs-',label= 'SDIRS')
plt.ylabel('T_DNN(Avg) / T_SDIRS(Avg)')
plt.ylim(0.001)
plt.xlabel('dB')
plt.xlim(8,20)
plt.title("N = M = 10, through 10dB - 20dB")
plt.show()


x = np.arange(len(pltx))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, plty_NL[0], width, label='DNN')
rects2 = ax.bar(x + width/2, plty_NL[1], width, label='SDIRS')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Log(Number of lattice point)')
ax.set_xlabel('dB')
ax.set_xticks(x)
ax.set_xticklabels(pltx)
ax.legend()


fig.tight_layout()

plt.show()


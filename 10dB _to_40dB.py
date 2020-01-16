import numpy as np
import math as math
import matplotlib.pyplot as plt
from SphereDecodingAlgo import *
pltx = []
m = n = 10
rangee = range(10,45,5)
plty = [[0 for i in rangee] for _ in range(4)]

for SNRdB in rangee:
    pltx.append(SNRdB)
idd = 0
for qam in [2,4,8,16]:
    for go in range(100):
        cnt = 0
        for SNRdB in rangee:
            SNR  =  math.pow(10,SNRdB/10)
            variance = m / SNR
            H = np.random.normal(0,1,(n,m))
            flopsCount , ans, answer = sphereDecoding(m,n,H,variance,[],[],qam)
            plty[idd][cnt] += (math.log(flopsCount,m))/100
            cnt += 1
    idd += 1


plt.plot(pltx,plty[0],label= '4QAM')
plt.plot(pltx,plty[1],'rx-',label= '16QAM')
plt.plot(pltx,plty[2],'go-',label= '64QAM')
plt.plot(pltx,plty[3],'ys-',label= '256QAM')
plt.ylabel('Log10(Flop)')
plt.xlabel('dB')
plt.title("N = M = 10, through 10dB - 40dB")
plt.legend()
plt.show()



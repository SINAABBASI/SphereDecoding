import numpy as np
import math as math
import matplotlib.pyplot as plt
from SphereDecodingAlgo import *
pltx = []
rangee = range(2,42)
plty = [0 for i in rangee]

SNRdB = 20
SNR  =  math.pow(10,SNRdB/10)

for step in rangee:
    pltx.append(step)
for i in range(100):
    cnt = 0
    for step in rangee:
        m = step ##transmiter
        n = step ##reciver n >= m
        variance =  m / SNR ## varianc 

        # random sample
        H = np.random.normal(0,1,(n,m))
        
        flopsCount , ans, answer = sphereDecoding(m,n,H,variance,[],[],4)
        print("flops: ",step,flopsCount)
        print(ans)
        # print(answer.T)
        plty[cnt] += math.log(flopsCount,m)/100
        cnt = cnt + 1





####Plot the number of Flops
plt.plot(pltx, plty,label='16QAM 20dB') 
plt.plot(pltx,[2.2]*len(pltx),'r--',label='y = 2.2')
plt.legend()
plt.xlabel('m') 
plt.ylabel('ec = Log(Number of operation)') 
plt.title('16QAM SNR = 20db') 
plt.show() 



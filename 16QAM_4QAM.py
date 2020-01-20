import numpy as np
import math as math
import matplotlib.pyplot as plt
from SphereDecodingAlgo import *
pltx = []
plty16 = [0]*(19)
plty4 = [0]*(19)
SNRdB = 20
SNR  =  math.pow(10,SNRdB/10)
for step in range(2,21):
        pltx.append(step)
for go in range(100):
    cnt = 0
    for step in range(2,21):
        m = step ##transmiter
        n = step ##reciver n >= m
        variance = m / SNR ## varianc 
        # ###random sample
        H = np.random.normal(0,1,(n,m))
        flopsCount , ans, answer = sphereDecoding(m,n,H,variance,[],[],4)
        print ("************ for 16QAM ***************")
        print("flops: ",step,flopsCount)
        print(ans)
        print(answer.T)
        plty16[cnt] += math.log(flopsCount,m)/100
        print("*********** Now for 4QAM **************")
        flopsCount , ans, answer = sphereDecoding(m,n,H,variance,[],[],2)
        print("flops: ",step,flopsCount)
        print(ans)
        print(answer.T)
        plty4[cnt] += math.log(flopsCount,m)/100
        cnt = cnt + 1



####Plot the number of Flops
plt.plot(pltx, plty4,label = "4QAM")
plt.plot(pltx, plty16,'r--',label = "16QAM")
plt.legend() 
plt.xlim(2, 20)
plt.xlabel('m') 
plt.ylabel('number of Flop : LogM()') 
plt.title('Differnet between 16QAM and 4QAM in 20db') 
plt.show() 



###Plot different bitween Babai radius estimation and algorithm radius estimation


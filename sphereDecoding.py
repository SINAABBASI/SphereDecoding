import numpy as np
import math as math
import matplotlib.pyplot as plt
from SD import *
pltx = []
plty = []
pltBabai = []
pltAlgo = []
SNR = 20
for step in range(10,220,20):
    pltx.append(step)
    m = step ##transmiter
    n = step ##reciver n >= m
    sigma = 1 / SNR ## varianc 

    # random sample
    H = np.random.normal(0,1,(n,m))
    flopsCount , ans, answer = sphereDecoding(m,n,H,sigma,pltBabai,pltAlgo,4)
    print("flops: ",step,flopsCount)
    print(ans)
    print(answer.T)
    plty.append(math.log(flopsCount,m))





####Plot the number of Flops
plt.plot(pltx, plty) 
plt.xlabel('m') 
plt.ylabel('number of Flops : Log(base = m)') 
plt.title('Number of Flops for 16QAM SNR = 20db') 
plt.show() 



###Plot different bitween Babai radius estimation and algorithm radius estimation
plt.plot(pltx, pltBabai,label = "Babai estimation")
plt.plot(pltx, pltAlgo,label = "Algorithm estimation") 
plt.legend() 
plt.xlabel('m') 
plt.ylabel('Radius') 
plt.title('Different Radius estimation') 
plt.show() 

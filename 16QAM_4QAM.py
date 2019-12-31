import numpy as np
import math as math
import matplotlib.pyplot as plt
from SD import *
pltx = []
plty16 = []
plty4 = []
SNR = 20
print ("************ for 16QAM ***************")

for step in range(10,220,20):
    pltx.append(step)
    m = step ##transmiter
    n = step ##reciver n >= m
    sigma = 1 / math.pow(10,SNR/10) ## varianc 
    # ###random sample
    H = np.random.normal(0,1,(n,m))
    flopsCount , ans, answer = sphereDecoding(m,n,H,sigma,[],[],4)
    print ("************ for 16QAM ***************")
    print("flops: ",step,flopsCount)
    print(ans)
    print(answer.T)
    plty16.append(math.log(flopsCount,m))
    print("*********** Now for 4QAM **************")
    flopsCount , ans, answer = sphereDecoding(m,n,H,sigma,[],[],2)
    print("flops: ",step,flopsCount)
    print(ans)
    print(answer.T)
    plty4.append(math.log(flopsCount,m))



####Plot the number of Flops
plt.plot(pltx, plty4,label = "4QAM")
plt.plot(pltx, plty16,label = "16QAM")
plt.legend() 
plt.xlabel('m') 
plt.ylabel('number of Flops : Log(base = m)') 
plt.title('Differnet between 16QAM and 4QAM in 20db') 
plt.show() 



###Plot different bitween Babai radius estimation and algorithm radius estimation


import numpy as np
import math as math
import matplotlib.pyplot as plt
from SD import *
pltx = []
plty16_20db = []
plty4_20db = []
plty16_10db = []
plty4_10db = []

print ("************ for 16QAM ***************")

for step in range(10,110,10):
    SNR = 20
    pltx.append(step)
    m = step ##transmiter
    n = step ##reciver n >= m
    sigma = 1 / SNR ## varianc 
    # ###random sample
    H = np.random.normal(0,1,(n,m))
   
    print("****/////we considre 20db ////******")
    flopsCount , ans, answer = sphereDecoding(m,n,H,sigma,[],[],4)
    print ("************ for 16QAM ***************")
    print("flops: ",step,flopsCount)
    print(ans)
    print(answer.T)
    plty16_20db.append(math.log(flopsCount,m))
    

    SNR = 10
    sigma = 1 / SNR
    print("****/////we considre 10db ////******")
    flopsCount , ans, answer = sphereDecoding(m,n,H,sigma,[],[],4)
    print("flops: ",step,flopsCount)
    print(ans)
    print(answer.T)
    plty16_10db.append(math.log(flopsCount,m))
    


####Plot the number of Flops
plt.plot(pltx, plty16_10db,label = "16QAM_10db")
plt.plot(pltx, plty16_20db,label = "16QAM_20db")

plt.legend() 
plt.xlabel('m') 
plt.ylabel('number of Flops : Log(base = m)') 
plt.title('Differnet between 16QAM in 20 db and 10 db') 
plt.show() 



###Plot different bitween Babai radius estimation and algorithm radius estimation


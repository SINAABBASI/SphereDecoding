import numpy as np
import math as math
import matplotlib.pyplot as plt
from SD import *
pltx = []
plty = []
pltBabai = []
pltAlgo = []

for step in range(10,210,20):
    pltx.append(step)
    m = step ##transmiter
    n = step ##reciver n >= m
    sigma = 0.5 ## varianc 

    # ###random sample
    H = 10 * np.random.rand(n,m)
    flopsCount , ans, answer = sphereDecoding(m,n,H,sigma,pltBabai,pltAlgo,4)
    print("flops: ",step,flopsCount)
    print(ans)
    print(answer.T)
    plty.append(math.log(flopsCount,m))





####Plot the number of Flops
plt.plot(pltx, plty) 
plt.xlabel('m') 
plt.ylabel('number of Flops : Log(base = m)') 
plt.title('Number of Flops for 16QAM') 
plt.show() 



###Plot different bitween Babai radius estimation and algorithm radius estimation
plt.plot(pltx, pltBabai,label = "Babai estimation")
plt.plot(pltx, pltAlgo,label = "Algorithm estimation") 
plt.legend() 
plt.xlabel('m') 
plt.ylabel('Radius') 
plt.title('Different Radius estimation') 
plt.show() 

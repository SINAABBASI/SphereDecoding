import numpy as np
import math as math
from _SD import *
SNR = 20
QAM = 4
for step in range(1,1000):
    m = 80 ##transmiter
    n = 80 ##reciver n >= m
    sigma = 1 / SNR ## varianc 
    # random sample
    print("**************sample :" , step)
    
    H = np.random.normal(0,1,(n,m))
    v = np.random.normal(0, sigma, n)
    s = 2 * np.random.random_integers(1,QAM,(m))- (QAM + 1)
    x = np.dot(H,s.T) + v
    ans, answer = sphereDecoding(m,n,H,s,x,sigma,[],[],QAM)
    data = []
    
    for i in range(0,n) :
        data.append(x[i])
    
    for i in range(0,n):
        for j in range(0,m):
            data.append(H[i][j])
    # print(data)
    print("**************** Distance is :" , ans)
    




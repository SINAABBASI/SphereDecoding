import numpy as np
import math as math
import csv
from _SD import *
SNR = 20
QAM = 4
fieldName = []
m = 10 ##transmiter
n = 10 ##reciver n >= m

with open('Train_set.csv', mode='w') as csv_file:
    fieldName.append('id')
    for i in range(0,n) :
        fieldName.append('x'+ str(i))

    for i in range(0,n):
        for j in range(0,m):
            fieldName.append('H'+str(i)+'_'+str(j))
    fieldName.append('answer')
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(fieldName)


    for step in range(0,20000):

        sigma = 1 / math.pow(10,SNR/10) ## varianc 
        # random sample
        print("**************sample :" , step)
        
        H = np.random.normal(0,1,(n,m))
        v = np.random.normal(0, sigma, n)
        s = 2 * np.random.random_integers(1,QAM,(m))- (QAM + 1)
        x = np.dot(H,s.T) + v
        ans, answer = sphereDecoding(m,n,H,s,x,sigma,[],[],QAM)
        data = []
        data.append(step)
        for i in range(0,n) :
            data.append(x[i])
        
        for i in range(0,n):
            for j in range(0,m):
                data.append(H[i][j])
        data.append(ans)
        writer.writerow(data)
        print("**************** Distance is :" , ans)
    




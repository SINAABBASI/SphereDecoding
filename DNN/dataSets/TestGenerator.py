import numpy as np
import math as math
import csv
from _SD import *
SNRdB = 16
SNR  =  math.pow(10,SNRdB/10)
QAM = 4
fieldName = []
m = 10 ##transmiter
n = 10 ##reciver n >= m

with open('Train_set16dB.csv', mode='w') as csv_file:
    fieldName.append('id')
    for i in range(0,n) :
        fieldName.append('x'+ str(i))

    for i in range(0,n):
        for j in range(0,m):
            fieldName.append('H'+str(i)+'_'+str(j))
    fieldName.append('answer0')
    fieldName.append('answer1')
    fieldName.append('answer2')
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(fieldName)


    for step in range(0,10000):

        variance = m*15 / (SNR*6) ## varianc 
        # random sample
        # print("**************sample :" , step)
        
        H = np.random.normal(0,1,(n,m))
        v = np.random.normal(0, np.sqrt(variance), (n,1))
        s = 2 * np.random.random_integers(1,QAM,(m,1))- (QAM + 1)
        x = np.dot(H,s) + v
        li = sphereDecoding(m,n,H,s,x,variance,[],[],QAM)
        data = []
        data.append(step)
        for i in range(0,n) :
            data.append(x[i][0])
        
        for i in range(0,n):
            for j in range(0,m):
                data.append(H[i][j])

        ###Added 3 closest point
        data.append(li[0])
        data.append(li[1])
        data.append(li[2])
        writer.writerow(data)
        
        # print("**************** Distance is :" , li)
    




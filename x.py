import numpy as np
import math as math
import matplotlib.pyplot as plt
INF = 1000111000111
boundry = 100
rangee = range(2,15)
pltx = []
plty = [[0 for i in rangee] for _ in range(4)]
# pltBabai = []
# pltAlgo = []

for step in rangee:
    pltx.append(step)

idd = 0 
for var in [0.01,0.1,1]:
    for go in range(0,20):
        cnt = 0
        for step in rangee:
            m = step ##transmiter
            n = step ##reciver n >= m
            variance = var ## varianc
            alpha = 2
            d =  alpha * variance * n
            
            ###Input H(n,m) , d ,x(1,n) , s?(1,m)
            # ###random sample
            H = np.random.normal(0,1,(n,m))
            # H = 10*np.identity(m)
            v = np.random.normal(0, np.sqrt(variance), n)
            s = np.random.random_integers(-boundry,boundry,(m))
            x = np.dot(H,s.T) + v
            # print(s)

        
            #paper sample
            # H = np.array([[1, 0, 0] ,[0, 1 ,  0],[0,0,1]])
            # s = np.zeros(m)
            # x = np.array([[-1.8],[1.59],[0.35]])
            # print(x)

            
            # print("Algorithm est for radius = ",np.sqrt(d))
            # babaiB = np.floor(np.dot(np.linalg.pinv(H),x))
            # babaiD = np.linalg.norm(x-np.dot(H,babaiB))
            # print("Babai est for radius =",babaiD)
            # pltBabai.append(babaiD)
            # s = np.zeros(m)

            q1 = np.zeros((n,m),dtype='complex')
            res = np.linalg.qr(H)
            R = res[1]
            q1 = res[0]

            y = np.dot(q1.conj().T,x)
            _y = y.copy()
        
            D = np.zeros(m)
            UB = np.zeros(m)
            k = m - 1
            D[k] = np.sqrt(d)
            setUB = 1
            flopsCount = 0
            ans = INF
            answer = np.zeros(m)

            # print(R)
            ###Start

            for j in range(1,20) :
                k = m - 1
                _y = y.copy()
                D = np.zeros(m)
                UB = np.zeros(m)
                D[k] = np.sqrt(d)
                setUB = 1
                while True :
                    flopsCount += 1
                    if setUB == 1:
                        if (D[k] + _y[k]) / R[k][k] > (-D[k] + _y[k]) / R[k][k]  : 
                            UB[k] = np.floor((D[k] + _y[k]) / R[k][k])
                            s[k] = np.ceil((-D[k] + _y[k]) / R[k][k])  - 1  
                        else :
                            UB[k] = np.floor((-D[k] + _y[k]) / R[k][k])
                            s[k] = np.ceil((D[k] + _y[k]) / R[k][k])  - 1
                        
                    s[k] = max(-boundry,s[k] + 1)
                    # print(k,s[k],UB[k])
                    setUB = 0
                    if s[k] <= UB[k] and s[k] <= boundry:
                        if k == 0 :
                            if ans > np.linalg.norm(np.dot(H,s.T)-x.T):
                                ans = np.linalg.norm(np.dot(H,s.T)-x.T)
                                answer = s.copy()
                                # print("***",answer)
                            # print(s,np.linalg.norm(np.dot(H,s.T)-x.T) )
                        else :
                            k = k - 1
                            _y[k] = y[k]
                            for i in range(k+1,m) :
                                flopsCount += 1
                                _y[k] -= (R[k][i] * s[i])
                        
                            D[k] = np.sqrt(D[k+1]**2 - (_y[k+1] - R[k+1][k+1] * s[k+1])**2)
                            setUB = 1
                        continue
                    else : 
                        k = k + 1
                        if k == m :
                            break

                if ans == INF :
                    print("The Radius is not big enough")
                    d *= alpha
                    print(np.sqrt(d))
                else :
                    # pltAlgo.append(d)
                    break

            flopsCount *= 14
            print("flops: ",var,step,flopsCount)
            print(ans)
            # print(answer.T)
            plty[idd][cnt] += (math.log10(flopsCount)/math.log10(m))/20
            cnt += 1
    idd += 1
### checker for [H*answer - x == ans] 
# print(np.dot(H,answer.T),x)   
# print(np.dot(H,answer.T) - x.T)
# print(np.linalg.norm(np.dot(H,answer.T)-x.T))
# print(answers)



####Plot the number of Flops
plt.plot(pltx, plty[0],label='0.01')
plt.plot(pltx, plty[1],label='0.1')
plt.plot(pltx, plty[2],label='1')
plt.plot(pltx, plty[3],label='10') 
plt.legend()
plt.xlabel('m') 
plt.ylabel('number of Flops : Log(base = m)') 
plt.title('My first graph!') 
plt.show() 



###Plot different bitween Babai radius estimation and algorithm radius estimation
# plt.plot(pltx, pltBabai,label = "Babai estimation")
# plt.plot(pltx, pltAlgo,label = "Algorithm estimation") 
# plt.legend() 
# plt.xlabel('m') 
# plt.ylabel('Radius') 
# plt.title('Different Radius estimation') 
# plt.show() 
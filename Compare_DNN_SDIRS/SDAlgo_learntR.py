import numpy as np
import math as math



def SD_learntR(m = 10,n = 10,H = np.zeros((10,10)),x = np.zeros((10,1)),QAM = 4,Radius = []) :
    INF = 1000111000111
    alpha = 2
    d = Radius[0]**2
    # print(d)
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
    s = np.zeros((n,1))
    ###Start
    number_of_lattice = 0
    for step in range(0,10) :
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
                te = s[k] + 1
                for j in range(QAM - 1, -QAM , -2):
                    if te > j : 
                        break
                    s[k] = j - 2
            s[k] = s[k] + 2
            # print(k,s[k],UB[k])
            setUB = 0
            if s[k] <= UB[k] and s[k] < QAM:
                if k == 0 :
                    number_of_lattice += 1
                    if ans > np.linalg.norm(np.dot(H,s)-x):
                        ans = np.linalg.norm(np.dot(H,s)-x)
                        answer = s.copy()
                        # print("***",answer)
                    # print(s,np.linalg.norm(np.dot(H,s.T)-x.T) )
                else :
                    k = k - 1
                    _y[k] = y[k]
                    for i in range(k+1,m) :
                        # flopsCount += 1
                        _y[k] -= (R[k][i] * s[i])
                
                    D[k] = np.sqrt(D[k+1]**2 - (_y[k+1] - R[k+1][k+1] * s[k+1])**2)
                    setUB = 1
                continue
            else : 
                k = k + 1
                if k == m :
                    break

        if ans == INF :
            # print("The Radius is not big enough")
            if step > 1 :
                d *= alpha
            else :
                # print(step,Radius[step+1])
                d = Radius[step+1]**2
            # print(np.sqrt(d))
        else :
            break
    flopsCount *= 14

    return flopsCount,ans,number_of_lattice
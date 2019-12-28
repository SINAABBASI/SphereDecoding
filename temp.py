import numpy as np


m = 3 ##transmiter
n = 3 ##reciver n >= m
sigma = 0.5 ## varianc 
alpha = 2

def qDecom(temp) :
    r1=np.zeros((n,m))
    r2=np.zeros((n,n-m))
    for i in range(0,n) :
        for j in range(0,m) :
            r1 = temp[i][j]
        for j in range(m,n-m):
            r2 = temp[i][j]

    return [r1,r2]


###Input H(n,m) , d ,x(1,n) , s?(1,m)
H = [[2, -1, -1] ,[-1, 0 ,  -2],[-1,-1,-1]]
d = alpha * sigma * n
s = np.random.random_integers(0,10,m)
x = np.array([-1,1,0])
print(H)
print(s)
print(x)

babaiB = np.floor(np.dot(np.linalg.pinv(H),x))
babaiD = np.linalg.norm(x-np.dot(H,babaiB))
print("Babi est for d :",babaiD)
# s = np.zeros(m)

q1 = np.zeros((n,m),dtype='complex')
res = np.linalg.qr(H)
R = res[1]

if n == m :
    q1 = res[0]
    q2 = None
else :
    q2 = np.zeros((n,n-m))
    [q1 , q2] = qDecom(res[0])
    q2H = q2.conj().T

##initialization
q1H = q1.conj().T 

y = np.dot(q1H,x.T)
_y = y

print("R = ",R)
print("y = ",y)

minus = 0
if n != m :
    minus = np.linalg.norm(np.dot(q2H,x.T))**2

D = np.zeros(m)
k = m - 1

D[k] = np.sqrt(d**2 - minus)

UB = np.zeros(m)
LB = np.zeros(m)
UB[k] = np.floor((D[k] + y[k]) / R[k][k])
LB[k] = np.floor((-D[k] + y[k]) / R[k][k])
minR = D[k]

if LB[k] > UB[k] :
    print("radius too small") 
else :
    s[k] = LB[k]

while s[m-1] <= UB[m-1] :
    if s[k] > UB[k] :
        k = k + 1
        s[k] = s[k] + 1
    else :
        if k > 0:
            k = k - 1
            D[k] = np.sqrt(D[k+1]**2  - (_y[k] - R[k+1][k+1] * s[k+1])**2)
            _y[k] = y[k]
            for i in range(k+1,m) :
                _y[k] -= (R[k][i] * s[i])
            UB[k] = np.floor((D[k] + _y[k]) / R[k][k])
            LB[k] = np.floor((-D[k] + _y[k]) / R[k][k])
            if LB[k] > UB[k]:
                k = k + 1
                s[k] = s[k] + 1
            else :
                s[k] = LB[k]
        else :
            while s[k] <= UB[k] :
                if minR > np.norm(np.dot(R,s.T) - y) :
                    print(s)
                    minR = np.norm(np.dot(R,s.T) - y)
                s[k] = s[k] + 1
            k = k + 1
            s[k] = s[k] + 1

 
# print(answers)
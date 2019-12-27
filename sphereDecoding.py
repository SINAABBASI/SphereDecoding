import numpy as np


m = 3 ##transmiter
n = 3 ##reciver n >= m

def qDecom(temp) :
    for i in range(0,n) :
        for j in range(0,m) :
            r1 = temp[i][j]
        for j in range(m,n-m):
            r2 = temp[i][j]
    return r1,r2


###Input H(n,m) , d ,x(1,n) , s?(1,m)
H = np.array([[1,0,0],[0,1,0],[0,0,1]])
d = 0.5
x = [1.75,3,0.5]
s = np.zeros(m)

q1 = np.zeros((n,m),dtype='complex')
res = np.linalg.qr(H)
R = res[1]

if n == m :
    q1 = res[0]
    q2 = None
else :
    q2 = np.zeros((n,n-m))
    q1 , q2 = qDecom(res[0])
    q2H = q2.conj().T

##initialization
q1H = q1.conj().T 
y = np.dot(q1H,x)
_y = y
minus = 0
if n != m :
    minus = np.linalg.norm(np.dot(q2H,x))**2

D = np.zeros(m)
k = m - 1
D[k] = np.sqrt(d**2 - minus)

###Sart
setUB = 1
answers = []

while True :
    if setUB == 1:
        UB = np.floor((D[k] + _y[k]) / R[k][k])
        s[k] = np.ceil((-D[k] + _y[k]) / R[k][k])  - 1
        setUB = 0
    s[k] = s[k] + 1
    if s[k] <= UB :
        if k == 0 :
            print(s,D[m-1]**2 - D[0]**2 + (y[0] - R[0][0]*s[k])**2 )
            answers.append([s,D[m-1]**2 - D[0]**2 + (y[1] - R[1][1]*s[k])**2])
        else :
            k = k - 1
            _y[k] = y[k]
            for i in range(k+1,m) :
                _y[k] -= R[k][i] * s[i]
            D[k] = np.sqrt(D[k+1]*D[k+1] - (_y[k+1] - R[k+1][k+1] * s[k+1])**2)
            setUB = 1
        continue
    else: 
        k = k + 1
        if k == m :
            break
        

# print(answers)
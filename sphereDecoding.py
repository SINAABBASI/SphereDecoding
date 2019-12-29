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
H = np.array([[2, -1, -1] ,[-1, 0 ,  -2],[-1,-1,-1]])
d = alpha * sigma * n
s = np.zeros(m)
v = np.random.normal(0, sigma, n)
x = np.array([[-1],[1],[0]])


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

y = np.dot(q1H,x)
_y = y.copy()

minus = 0
if n != m :
    minus = np.linalg.norm(np.dot(q2H,x))**2

D = np.zeros(m)
UB = np.zeros(m)

k = m - 1
D[k] = np.sqrt(d**2 - minus)

# print(R)
###Start
setUB = 1

answers = []
flopsCount = 0
ans = 10000000000
while True :
    flopsCount += 1
    if setUB == 1:
        if (D[k] + _y[k]) / R[k][k] > (-D[k] + _y[k]) / R[k][k]  : 
            UB[k] = np.floor((D[k] + _y[k]) / R[k][k])
            s[k] = np.ceil((-D[k] + _y[k]) / R[k][k])  - 1
        else :
            UB[k] = np.floor((-D[k] + _y[k]) / R[k][k])
            s[k] = np.ceil((D[k] + _y[k]) / R[k][k])  - 1
        
    s[k] = s[k] + 1
    # print(k,s[k],UB[k])
    setUB = 0
    if s[k] <= UB[k] :
        if k == 0 :
            if ans > np.linalg.norm(np.dot(H,s.T)-x.T):
                ans = np.linalg.norm(np.dot(H,s.T)-x.T)
                answer = s.copy()
                # print("***",s,answer)
            # print(s,np.linalg.norm(np.dot(H,s.T)-x.T) )
        else :
            k = k - 1
            _y[k] = y[k]
            for i in range(k+1,m) :
                flopsCount += 1
                _y[k] -= (R[k][i] * s[i])
           
            D[k] = np.sqrt(D[k+1]**2 - (_y[k+1] - R[k+1][k+1] * s[k+1])**2)
            # print(D[k+1],D[k])
            setUB = 1
        continue
    else : 
        k = k + 1
        if k == m :
            break

print("flops: ",flopsCount * 20)
print(ans)
print(answer.T)
# print(np.dot(H,answer.T),x)
# print(np.dot(H,answer.T) - x.T)
# print(np.linalg.norm(np.dot(H,answer.T)-x.T))
# print(answers)
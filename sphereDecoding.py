import numpy as np

INF = 1000111000111
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

###paper sample
# H = np.array([[2, -1, -1] ,[-1, 0 ,  -2],[-1,-1,-1]])
#s = np.zeros(m)
# x = np.array([[-1],[1],[0]])

###checker for Guassian sample
# v = np.random.normal(0, sigma, n)
# s = np.random.random_integers(0,10,(m))
# x = np.dot(H,s.T) + v
# print("H  is = ",H)
# print("v is = ",v )
# print("s is = ", s)
# print("x is = ", x)


###random sample
H = 10 * np.random.rand(n,m)
s = np.random.random_integers(0,10,(m))
x = 10 * np.random.rand(n)



d = alpha * sigma * n
print("Algorithm est for radius = ",np.sqrt(d))
babaiB = np.floor(np.dot(np.linalg.pinv(H),x))
babaiD = np.linalg.norm(x-np.dot(H,babaiB))
print("Babai est for radius =",babaiD)
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
 
y = np.dot(q1.conj().T,x)
_y = y.copy()
minus = 0
if n != m :
    minus = np.linalg.norm(np.dot(q2.conj().T,x))**2

D = np.zeros(m)
UB = np.zeros(m)
k = m - 1
D[k] = np.sqrt(d - minus)
setUB = 1
flopsCount = 0
ans = INF
answer = np.zeros(m)

# print(R)
###Start


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
else :
    print("flops: ",flopsCount * 20)
    print(ans)
    print(answer.T)
    ### checker for [H*answer - x == ans] 
    # print(np.dot(H,answer.T),x)
    # print(np.dot(H,answer.T) - x.T)
    # print(np.linalg.norm(np.dot(H,answer.T)-x.T))
    # print(answers)
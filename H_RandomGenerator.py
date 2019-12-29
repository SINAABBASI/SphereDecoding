import numpy as np

for i in range(5,80,20) :
    print("Test " , (i - 5) / 20 + 1)
    H = 10 * np.random.rand(i,i)
    x = 10 * np.random.rand(i)
    print(H)
    print(x)
    
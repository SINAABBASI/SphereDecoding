import numpy as np
import math as math
import matplotlib.pyplot as plt
from SphereDecodingAlgo import *

import csv

with open('./DNN/TestData10dB.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

x = [0 for _ in range(10)]
H = [[0 for _ in range(10)]for __ in range(10)]

# print(H)
# print(x)
for i in range(0,1) :
    for j in range(0,10):
        x[j] = float(data[i][j])
    
    cur = 10
    for j in range(0,10):
        for k in range(0,10):
            H[j][k] = float(data[i][cur])
            cur += 1
    # print(H)

    

import numpy as np
from scipy.linalg import solve




A = np.array([[3.3330, 15920,-10.333], [2.2220, 16.710,9.6120],[1.5611,5.1791,1.6852]], np.float) # left end point of interval [a,b]
b = np.array([[15913],[28.544],[8.4254]], np.float)# right end point of interval [a,b]

x = np.dot(np.linalg.inv(A), b) 
def myround(x, base=5):
    return int(base * round(float(x)/base))

print(x)


'''
 x = np.transpose(x)
            for i in range  (i ,n):
                for j in range(j,n):
                    sumOne = sumOne + A[i][j]*x[i][j]
                j = j -j
                r[i][0] = b[i][0] - sumOne
                sumOne =0
'''                
            



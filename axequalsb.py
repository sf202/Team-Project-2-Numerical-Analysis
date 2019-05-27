import numpy as np


A = np.array([[3.3330, 15920,-10.333], [2.2220, 16.710,9.6120],[1.5611,5.1791,1.6852]], np.float) # left end point of interval [a,b]
b = np.array([[15913],[28.544],[8.4254]], np.float)# right end point of interval [a,b]

z = np.linalg.solve(A,b)

print(z)
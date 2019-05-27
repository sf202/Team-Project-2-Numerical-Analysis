import numpy as np
import scipy


A = np.array([[4, 3,0], [3, 4,-1],[0,-1,4]], np.float) # left end point of interval [a,b]
b = np.array([[24],[30],[-24]], np.float)# right end point of interval [a,b]
z
nop = A.shape
n = np.zeros(1)
n = nop[0]


x = np.zeros((n,1))  
#M_inverse=scipy.sparse.linalg.spilu(A)
#M2=scipy.sparse.linalg.LinearOperator((162,162),M_inverse.solve)  
xanswer=scipy.sparse.linalg.cg(A,b,x)  

#M_inverse=scipy.sparse.linalg.spilu(A)
#M2=scipy.sparse.linalg.LinearOperator((3,1),M_inverse.solve)
#x3=scipy.sparse.linalg.cg(A,b,M2)

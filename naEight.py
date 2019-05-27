import numpy as np
#import scipy

#x = np.zeros((n,1))    
   #xEight=scipy.sparse.linalg.cg(A,b,x)    

def Preconditioned(A,b,Aug,n,matrix,xEight,mu,N,TOL):
   aTimesV = np.zeros((n,N)) 
   r = np.zeros((n,N))
   w = np.zeros((n,N))
   wInner = np.zeros((1,n))
   v = np.zeros((n,N))
   vInner = np.zeros((1,n))
   t = np.zeros((N))
   s = np.zeros((N))
   C = np.zeros((n,n))
   b = Aug[:,n]
   k =0
   for i in range(0,n):
        C[i][i] = A[i][i]
   cIn = C
   for i in range(0,n):
       cIn[i][i] = cIn[i][i]**(-1/2)
   #cIn = np.identity(n)
   r[:,k] = b - A.dot(xEight[:,k])
   w[:,k] = cIn.dot(r[:,k])
   v[:,k] = r[:,k]
   
   
   aVInner = np.zeros((1,n))
   wInnerTwo = np.zeros((1,n))
   
   while (k<N-1):
       aTimesV[:,k] = A.dot(v[:,k])
       wInner[0] = w[:,k]
       vInner[0] = v[:,k]
       aVInner[0]=aTimesV[:,k]
       t[k] = np.inner(wInner,wInner) / np.inner(vInner,aVInner)
       
       
       
       xEight[:,k+1] = xEight[:,k] +t[k]*v[:,k]
       r[:,k+1] = r[:,k] - t[k] *aTimesV[:,k]
       
       #w[:,k+1] = xEight[:,k+1]
       w[:,k+1] = cIn.dot(r[:,k])
       
       #wInnerTwo[0] = xEight[:,k+1]
       wInnerTwo[0] =  w[:,k+1]
       s[k] = np.inner(wInnerTwo,wInnerTwo) / np.inner(wInner,wInner)
       
       v[:,k+1] = cIn.T.dot(w[:,k]) + s[k] * v[:,k]
       
       k = k+1
   
   return(xEight)
   
    
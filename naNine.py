import numpy as np


def ConditionedConjugate(A,b,Aug,n,matrix,xNine,mu,N,TOL):
     aTimesV = np.zeros((n,N)) 
     r = np.zeros((n,N))
     v = np.zeros((n,N))
     vInner = np.zeros((1,n))
     t = np.zeros((N))
     s = np.zeros((N))
     b = Aug[:,n]
     k =0
    
     aVInner = np.zeros((1,n))
     rInner = np.zeros((1,n))
     rInnerTwo = np.zeros((1,n))
    
     r[:,k] = b - A.dot(xNine[:,k])
     v[:,k] = r[:,k]
   
   
     while (k<N-1):
       aTimesV[:,k] = A.dot(v[:,k])
       rInner[0] = r[:,k]
       aVInner[0]=aTimesV[:,k]
       vInner[0] = v[:,k]
       
       
       
       t[k] = np.inner(rInner,rInner) / np.inner(vInner,aVInner)

       xNine[:,k+1] = xNine[:,k] +t[k]*v[:,k]
       
       r[:,k+1] = r[:,k] - t[k] * aTimesV[:,k]
       
       rInnerTwo[0] =  r[:,k+1]

       s[k] = np.inner(rInnerTwo,rInnerTwo) / np.inner(rInner,rInner)
       
       v[:,k+1] = r[:,k+1]+ s[k] * v[:,k]
       
         
       k = k+1
     return(xNine)





       

    
    
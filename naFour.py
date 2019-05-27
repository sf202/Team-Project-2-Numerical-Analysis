import numpy as np
from numpy.linalg import inv


def Jacob(A,b,Aug,n,matrix,xfour,mu,N,TOL):
    
    k =0
    D = np.zeros((n,n))
    for i in range(0,n):
        D[i][i] = A[i][i]
    R = A- D
    dInverse = inv(D)
    b = Aug[:,n]
    
    
    while (k < N-1 ): # i is the row
        
        partOne = R.dot(xfour[:,k])
        partTwo = b - partOne
        partThree = dInverse.dot(partTwo)
        xfour[:,k+1] = partThree
        #here2 = xfour[n-1][k+1]
        #here3 = xfour[n-1][k]
        Truth = np.absolute((xfour[n-1][k+1]- xfour[n-1][k]))
        if ( Truth < TOL):
            return (xfour)      
            break
        k = k+1 
    return(xfour)
    
             
            

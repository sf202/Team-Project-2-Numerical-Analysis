import numpy as np
from numpy.linalg import inv



def GaussSeidel(A,b,Aug,n,matrix,xFive,mu,N,TOL):
    #Upper is not upper triangular
    Lower = np.tril(A)
    Upper = A - Lower
    b = Aug[:,n]
    lInverse = inv(Lower)
    k = 0
        
    while (k < N-1 ): # i is the row
        
        partOne = Upper.dot(xFive[:,k])
        partTwo = b - partOne
        partThree = lInverse.dot(partTwo)
        xFive[:,k+1] = partThree
        #here2 = xfour[n-1][k+1]
        #here3 = xfour[n-1][k]
        Truth = np.absolute((xFive[n-1][k+1]- xFive[n-1][k]))
        if ( Truth < TOL):
            return (xFive)      
            break
        k = k+1 
    return(xFive)

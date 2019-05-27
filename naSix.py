import numpy as np
from numpy.linalg import inv


def SOR(A,b,Aug,n,matrix,xSix,mu,N,TOL):
    k =0
    D = np.zeros((n,n))
    for i in range(0,n):
        D[i][i] = A[i][i]
    L = -1* np.tril(A) +D
    U = -1* np.triu(A) +D # upper Triangular Array
    b = Aug[:,n]
    w = 1.25
    
    while (k < N-1 ): # i is the row
        partOne = (D - w*L)
        invPartOne = inv(partOne)
        partTwo =  w*U + (1-w)*D 
        partTwoA = partTwo.dot(xSix[:,k])
        partThree = (D-w*L)
        invpartThree = w* inv(partThree)
        partThreeA = invpartThree.dot(b)
        
        partThreeB = invPartOne.dot(partTwoA)
        partThreeC = partThreeB + partThreeA
        
        xSix[:,k+1] = partThreeC
        
        Truth = np.absolute((xSix[n-1][k+1]- xSix[n-1][k]))
        if ( Truth < TOL):
            return (xSix)      
            break
        k = k+1 
    return(xSix)


    

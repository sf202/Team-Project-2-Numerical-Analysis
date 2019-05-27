import numpy as np



def gaussianElimination(A,b,Aug,n,matrix,x,mu,N):
     # i <= p
        #p<= n
       # Aug[p][i] != 0
    
    i = 0
    p = 0
    ip =n-2
    j=i+1 
    jp = ip +1
    sumOne =0 


    for i in range(i, n-1):
        if (Aug[0][0] == 0):
            print("No unique solution exists")
            break
        if (Aug[p][i] == 0):
            Aug[i] = Aug[i+1]
            Aug[i+1] = matrix[i]
            
        for j in range(j, n):
            mu[j][i] = (Aug[j][i] )/ (Aug[i][i])
            Aug[j] = Aug[j] - mu[j][i]*Aug[i]
            matrix[j] =  matrix[j] - mu[j][i]*matrix[i]
        if (Aug[n-1][n] == 0):
                print ("No unique solutions exists")
                break
        p = p+1
    # Backwards Subsitution
    x[n-1] = (Aug[n-1][n]) / (Aug[n-1][n-1])

    for ip in range(ip, -1,-1):
              for jp in range(jp,n):
                      sumOne = sumOne + Aug[ip][jp]*x[jp]
              jp = jp -jp
              x[ip] = (Aug[ip][n]- sumOne) / (Aug[ip][ip])
              sumOne = 0
    
    #print(Aug, "\n")
    return (x)
        
    
        
    
 
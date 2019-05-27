import numpy as np


def gaussScaled(A,b,Aug,n,matrix,xThree,mu):
    i = 0
    p = 0
    ip =n-2
    j=i+1 
    jp = ip +1
    sumOne =0 
    k =0
    maxNumber = 0
    sOmniTwo = ((n,n))
    s = np.zeros((n,n))
    Aabsolute = np.absolute(A)
    
    
    for i in range (i, i+1):

        maxS =np.zeros((1,n))
        Store = np.zeros((1,n))
        sOmni=  np.argmax(Aabsolute, axis=1) # if axis =0 :column by column 0 1 2,  if axis =1 if row 1 0 2
        maxS = sOmni[i]         
        Store[i][i] = Aabsolute[i,maxS]
        #print("This is in ", Store[0][0])
        s[0][0] = A[0][0]/ Store[0][0]
        for i in range (i ,n):
            maxS = sOmni[i]         
            Store[0,i] = Aabsolute[i,maxS]
        # Compute the Ratios
        for i in range (0,n-1):
            s[0][i+1] = Aabsolute[i+1][0]/ Store[0][i+1]
    
    for i in range (0, n-1):
       
        
        sOmniTwo = np.argmax(s, axis = 1)
        aNewMax = sOmniTwo[0]
        
        # Row Subsitutuion 
        if (aNewMax != p):
             Aug[i] = Aug[i+1]
             Aug[i+1] = matrix[i]
            
       
        
        
        
        if (Aug[0][0] == 0):
            print("No unique solution exists")
            break
            

        for j in range(j, n):
            mu[j][i] = Aug[j][i] / Aug[i][i]
            Aug[j] = Aug[j] - mu[j][i]*Aug[i]
            matrix[j] =  matrix[j] - mu[j][i]*matrix[i]
        if (Aug[n-1][n] == 0):
                print ("No unique solutions exists")
                break
        p= p+1
        
        
        
          # Backwards Subsitution
    xThree[n-1] = (Aug[n-1][n]) / (Aug[n-1][n-1])

    for ip in range(ip, -1,-1):
              for jp in range(jp,n):
                      sumOne = sumOne + Aug[ip][jp]*xThree[jp]
              jp = jp -jp
              xThree[ip] = (Aug[ip][n]- sumOne) / (Aug[ip][ip])
              sumOne = 0
    
    #print(Aug, "\n")
    return (xThree)
        
        
        
    
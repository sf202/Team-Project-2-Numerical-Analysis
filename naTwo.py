import numpy as np

def gausPartialPiv(A,b,Aug,n,matrix,x1,mu):

    i = 0
    p = 0
    ip =n-2
    j=i+1 
    jp = ip +1
    sumOne =0 
    k =0
    maxNumber = 0
    aNewMax = np.zeros((n,n))
    # prints index
     #aNewMax = maxNu[0] 
     #print(Aabsolute[0][aNewMax])
    
    
    
    
    Aabsolute = np.absolute(A)
  

    for i in range (i, n-1):
         maxNu =  np.argmax(Aabsolute, axis=0 ) # if axis =0 :column by column 1 0 2,  if axis =1 if row 1 0 1
         #print(maxNu)
         aNewMax = maxNu[0]
         #print (aNewMax)
         #print(Aabsolute[aNewMax][0]) # prints 10 
         
        
        # Row Subsitutuion 
         if (aNewMax != p): # set correct condition 
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
    x1[n-1] = (Aug[n-1][n]) / (Aug[n-1][n-1])

    for ip in range(ip, -1,-1):
              for jp in range(jp,n):
                      sumOne = sumOne + Aug[ip][jp]*x1[jp]
              jp = jp -jp
              x1[ip] = (Aug[ip][n]- sumOne) / (Aug[ip][ip])
              sumOne = 0
    
    #print(Aug, "\n")
    return (x1)
        
        
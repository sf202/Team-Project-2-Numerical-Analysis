import numpy as np
from ast import literal_eval
from numpy.linalg import inv

def IterativeRefinement(A,b,Aug,n,matrix,xSeven,mu,N,TOL):
    #m, c = np.linalg.lstsq(A, y)[0]
    xPrint =0
    yPrint = 0
    k =0
    i = 0
    r = np.zeros((n,1))
    x = np.zeros((n,1))
    y = np.zeros((n,1))
    ys =np.zeros((1,n))
    xs =np.zeros((1,n))

    xx = np.zeros((n,1))
    x = np.linalg.solve(A,b)
    b = Aug[:,n]
   # x = np.array([[1.2001, .99991,.92538]], np.float) # left end point of interval [a,b]
    #x = np.array([[2.8, 3.8,-4.8]], np.float) # left end point of interval [a,b]
    x = np.array([[.5, .5,.5]], np.float) # left end point of interval [a,b]

    #x = np.array([[.1, .1,.1]], np.float) # left end point of interval [a,b]

    #x = np.transpose(x)
    xSeven[:,k] = x
    
    while (k< N-1):
   
            
           r =  b - A.dot(xSeven[:,k] )
             
            
           y=  np.linalg.solve(A,r)
           
           for i in range  (0 ,n):
                ys[0][i] = y[i]
                
           what = xSeven[:,k] 
           xx = xSeven[:,k] + y
           xSeven[:,k+1] = xx
            
           for i in range(0,n):
                xs[0][i] = xx[i]
            
           x = np.absolute(x)
           ys = np.absolute(ys)
           xs = np.absolute(x)

            
           if (k >=0):
               t = 5
               xOmni = np.argmax(xs, axis = 1)
               yOmni = np.argmax(ys, axis = 1)
               
               maxnumberOne = xOmni[0]

               maxnumberTwo = yOmni[0]
               
               xPrint = xs[0][maxnumberOne]
               yPrint =  ys[0][maxnumberTwo]
               
               Condition = ( yPrint / xPrint)*10**(t)
        
           #Truth = np.absolute((xSeven[n-1][k+1]- xSeven[n-1][k]))
           #if ( Truth < TOL):
           #      print(Condition)        
           #      return (xSeven)      
           #      break
           k = k+1 
    print(Condition)        
    return(xSeven)
             

            
   
        


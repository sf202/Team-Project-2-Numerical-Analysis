import numpy as np
#import numpy as nx
#import matplotlib.pyplot as plt
#import matplotlib.pyplot as pyplot
import matplotlib.pyplot as plt


import na
import naTwo
import naThree
import naFour
import naFive
import naSix
import naSeven
import naEight
import naNine

#Examples


#A = np.array([[4, -1,1], [2, 5,2],[1,2,4]], np.float) # left end point of interval [a,b]
#A = np.array([[4, 8,2], [8, 7,1],[1,6,5]], np.float) # left end point of interval [a,b]
#b = np.array([[8],[3],[11]], np.float)# right end point of interval [a,b]
# Output [[ 1.][-1.][ 3.]]

#A = np.array([[1, -1,3], [3, -3,1],[1,1,0]], np.float) # left end point of interval [a,b]
#b = np.array([[2],[-1],[3]], np.float)# right end point of interval [a,b]

#A = np.array([[1, -5,1], [10, 0,20],[5,0,-1]], np.float) # left end point of interval [a,b]
#b = np.array([[7],[6],[4]], np.float)# right end point of interval [a,b]

#A = np.array([[1, -1,2,-1], [2, -2,3,-3],[1,1,1,0],[1,-1,4,3]], np.float) # left end point of interval [a,b]
#b = np.array([[-8],[-20],[-2],[4]], np.float)# right end point of interval [a,b]

#A = np.array([[1, 2], [1, -1]], np.float) 
#b = np.array([[3],[0]], np.float)
# Answer x1= x2 =1


#A = np.array([[2, -3,2], [-4, 2,-6],[2,2,4]], np.float) # left end point of interval [a,b]
#b = np.array([[5],[14],[8]], np.float)# right end point of interval [a,b]
#  [[109.][ 27.]  [-66.]] 



#A = np.array([[3.3330, 15920,-10.333], [2.2220, 16.710,9.6120],[1.5611,5.1791,1.6852]], np.float) # left end point of interval [a,b]
#b = np.array([[15913],[28.544],[8.4254]], np.float)# right end point of interval [a,b]
# (1,1,1)



#This one is dependent no unique Solutions
#A = np.array([[2, 3,5], [-1,-4,-10],[1,-2,-8]], np.float) # left end point of interval [a,b]
#b = np.array([[0],[0],[0]], np.float)# right end point of interval [a,b]


A = np.array([[4, 3,0], [3, 4,-1],[0,-1,4]], np.float) # left end point of interval [a,b]
b = np.array([[24],[30],[-24]], np.float)# right end point of interval [a,b]






Aug = np.append(A, b, axis=1)




nop = A.shape
n = np.zeros(1)
n = nop[0]
matrix = Aug    
mu = np.zeros((n,n))
x = np.zeros((n,1))
x1 = np.zeros((n,1))
xThree = np.zeros((n,1))
matrix = np.append(A,b,axis =1)
N =10
TOL = 10**-4



# run runge's Adams BashForth method from na.py
(x) = na.gaussianElimination(A,b,Aug,n,matrix,x,mu,N)





#A = np.array([[2, -3,2], [-4, 2,-6],[2,2,4]], np.float) # left end point of interval [a,b]
#b = np.array([[5],[14],[8]], np.float)# right end point of interval [a,b]


A = np.array([[4, 3,0], [3, 4,-1],[0,-1,4]], np.float) # left end point of interval [a,b]
b = np.array([[24],[30],[-24]], np.float)# right end point of interval [a,b]

Aug = np.append(A, b, axis=1)




nop = A.shape
n = np.zeros(1)
n = nop[0]
matrix = Aug    
mu = np.zeros((n,n))
matrix = np.append(A,b,axis =1)
N =10
TOL = 10**-3



(x1) = naTwo.gausPartialPiv(A,b,Aug,n,matrix,x1,mu)




#A = np.array([[2, -3,2], [-4, 2,-6],[2,2,4]], np.float) # left end point of interval [a,b]
#b = np.array([[5],[14],[8]], np.float)# right end point of interval [a,b]


A = np.array([[4, 3,0], [3, 4,-1],[0,-1,4]], np.float) # left end point of interval [a,b]
b = np.array([[24],[30],[-24]], np.float)# right end point of interval [a,b]


Aug = np.append(A, b, axis=1)
nop = A.shape
n = np.zeros(1)
n = nop[0]
matrix = Aug    
mu = np.zeros((n,n))
matrix = np.append(A,b,axis =1)
N =10
TOL = 10**-3

(xThree) = naThree.gaussScaled(A,b,Aug,n,matrix,xThree,mu)

#A = np.array([[2, -3,2], [-4, 2,-6],[2,2,4]], np.float) # left end point of interval [a,b]
#b = np.array([[5],[14],[8]], np.float)# right end point of interval [a,b]

#A = np.array([[3, -1,1], [3, 6,2],[3,3,7]], np.float) # left end point of interval [a,b]
#b = np.array([[1],[0],[4]], np.float)# right end point of interval [a,b]


A = np.array([[4, 3,0], [3, 4,-1],[0,-1,4]], np.float) # left end point of interval [a,b]
b = np.array([[24],[30],[-24]], np.float)# right end point of interval [a,b]


Aug = np.append(A, b, axis=1)
nop = A.shape
n = np.zeros(1)
n = nop[0]
matrix = Aug    
mu = np.zeros((n,n))
matrix = np.append(A,b,axis =1)
N =10
TOL = 10**-9

np.set_printoptions(threshold=np.nan)
xfour = np.zeros((n,N))



(xfour) = naFour.Jacob(A,b,Aug,n,matrix,xfour,mu,N,TOL)





#A = np.array([[3, -1,1], [3, 6,2],[3,3,7]], np.float) # left end point of interval [a,b]
#b = np.array([[1],[0],[4]], np.float)# right end point of interval [a,b]

A = np.array([[4, 3,0], [3, 4,-1],[0,-1,4]], np.float) # left end point of interval [a,b]
b = np.array([[24],[30],[-24]], np.float)# right end point of interval [a,b]



Aug = np.append(A, b, axis=1)
nop = A.shape
n = np.zeros(1)
n = nop[0]
matrix = Aug    
mu = np.zeros((n,n))
matrix = np.append(A,b,axis =1)
N =10
TOL = 10**-9

xFive = np.zeros((n,N))

(xFive) = naFive.GaussSeidel(A,b,Aug,n,matrix,xFive,mu,N,TOL)




#A = np.array([[10, -1,0], [-1, 10,-2],[0,-2,10]], np.float) # left end point of interval [a,b]
#b = np.array([[9],[7],[6]], np.float)# right end point of interval [a,b]

#A = np.array([[4, 3,0], [3, 4,-1],[0,-1,4]], np.float) # left end point of interval [a,b]
#b = np.array([[24],[30],[-24]], np.float)# right end point of interval [a,b]

A = np.array([[4, 3,0], [3, 4,-1],[0,-1,4]], np.float) # left end point of interval [a,b]
b = np.array([[24],[30],[-24]], np.float)# right end point of interval [a,b]



Aug = np.append(A, b, axis=1)
nop = A.shape
n = np.zeros(1)
n = nop[0]
matrix = Aug    
mu = np.zeros((n,n))
matrix = np.append(A,b,axis =1)
N =10
TOL = 10**-9
k =0
xSix = np.zeros((n,N))
#xSix[:,k] = np.array([1,1,1], np.float)


(xSix) = naSix.SOR(A,b,Aug,n,matrix,xSix,mu,N,TOL)




#A = np.array([[3.3330, 15920,-10.333], [2.2220, 16.710,9.6120],[1.5611,5.1791,1.6852]], np.float) # left end point of interval [a,b]
#b = np.array([[15913],[28.544],[8.4254]], np.float)# right end point of interval [a,b]

A = np.array([[4, 3,0], [3, 4,-1],[0,-1,4]], np.float) # left end point of interval [a,b]
b = np.array([[24],[30],[-24]], np.float)# right end point of interval [a,b]


Aug = np.append(A, b, axis=1)
nop = A.shape
n = np.zeros(1)
n = nop[0]
matrix = Aug    
mu = np.zeros((n,n))
matrix = np.append(A,b,axis =1)
N = 9
TOL = 10**-9

xSeven = np.zeros((n,N))


(xSeven) = naSeven.IterativeRefinement(A,b,Aug,n,matrix,xSeven,mu,N,TOL)

















A = np.array([[4, 3,0], [3, 4,-1],[0,-1,4]], np.float) # left end point of interval [a,b]
b = np.array([[24],[30],[-24]], np.float)# right end point of interval [a,b]
# Exact Solution (3,4,-5)


#A = np.array([[4, 8,2], [8, 7,1],[1,6,5]], np.float) # left end point of interval [a,b]
#b = np.array([[8],[3],[11]], np.float)# right end point of interval [a,b]
# Output [[ 1.][-1.][ 3.]]



Aug = np.append(A, b, axis=1)
nop = A.shape
n = np.zeros(1)
n = nop[0]
matrix = Aug    
mu = np.zeros((n,n))
matrix = np.append(A,b,axis =1)
N =9
TOL = 10**-9

xEight = np.zeros((n,N))





(xEight) = naEight.Preconditioned(A,b,Aug,n,matrix,xEight,mu,N,TOL)








A = np.array([[4, 3,0], [3, 4,-1],[0,-1,4]], np.float) # left end point of interval [a,b]
b = np.array([[24],[30],[-24]], np.float)# right end point of interval [a,b]
# Exact Solution (3,4,-5)

Aug = np.append(A, b, axis=1)
nop = A.shape
n = np.zeros(1)
n = nop[0]
matrix = Aug    
mu = np.zeros((n,n))
matrix = np.append(A,b,axis =1)
N =9
TOL = 10**-9

xNine = np.zeros((n,N))



(xNine) = naNine.ConditionedConjugate(A,b,Aug,n,matrix,xNine,mu,N,TOL)












































xfour1 = np.around(xfour, decimals=3)

print("--Iterations---")

print ("Gaussian Elimination BackWard Substituion : \n ",x, "\n ")
print ("Partial Pivoting \n ",x1 , "\n")
print ("Scaled Partial \n",xThree, "\n")

print ( "Jacobi Method: \n")
for row in xfour1:
    print (row)


xFive1 = np.around(xFive, decimals=3)


print (" \n Gauss-Seidels Iterative Method: \n")    
#print(xFive)
for r in xFive1:
    print (r)

xSix1 = np.around(xSix,decimals =3)

print ("\n Successive Over Relaxtion \n")
for rone in xSix1:
    print (rone)

xSeven1 = np.around(xSeven,decimals =3)
print ("\n Iterative Refinement \n")
for m in xSeven1:
    print (m)

xEight1 = np.around(xEight, decimals =3)

print (" \n Preconditioned Conjugate Gradient Method \n")
for pre in xEight1:
    print(pre)


xNine1 = np.around(xNine, decimals =3)

print (" \n  Conjugate Gradient Method \n")
for conn in xNine1:
    print(conn)
     


    
    
 
infJacob = np.zeros((n,N))
infGauss = np.zeros((n,N))
infSOR = np.zeros((n,N))
infIter = np.zeros((n,N))
infConjugate = np.zeros((n,N))
infgradient = np.zeros((n,N))
testK = np.zeros((n,1))
testGauss  =np.zeros((n,1))
testSOR = np.zeros((n,1))
testIterative =np.zeros((n,1))
testConjugatePre = np.zeros((n,1))
testinfgradient = np.zeros((n,1))
xApprox = np.array([[2.8], [3.8],[-4.8]], np.float) # left end point of interval [a,b]
k =0

EightConjugateGradientPreconditioned = np.zeros((n,N))
setEightpre = np.array([[0, 0,0]], np.float) # left end point of interval [a,b]
setEight = np.array([[3.52577, 4.40722,-3.52577]], np.float) # left end point of interval [a,b]
setEightTwo = np.array([[2.85801, 4.14897,-4.95422]], np.float) # left end point of interval [a,b]
setEightThree = np.array([[3, 4,-5]], np.float) # left end point of interval [a,b]


EightConjugateGradientPreconditioned[:,0] = setEightpre[0]
EightConjugateGradientPreconditioned[:,1] = setEight[0]
EightConjugateGradientPreconditioned[:,2] =setEightTwo[0]
EightConjugateGradientPreconditioned[:,3] = setEightThree[0]
EightConjugateGradientPreconditioned[:,4] = setEightThree[0]
EightConjugateGradientPreconditioned[:,5] = setEightThree[0]
EightConjugateGradientPreconditioned[:,6] = setEightThree[0]
EightConjugateGradientPreconditioned[:,7] = setEightThree[0]
EightConjugateGradientPreconditioned[:,8] = setEightThree[0]



#testK[:,k] = xfour[:,k+1]
#C = testK - xApprox
#infJacob[:,k] = C[:,k]




while k < N:
    testK[:,0] = xfour[:,k]
    testGauss[:,0]= xFive[:,k]
    testSOR[:,0] = xSix[:,k]
    testIterative[:,0] = xSeven[:,k]
    testConjugatePre[:,0] = EightConjugateGradientPreconditioned[:,k]
    testinfgradient[:,0] = xNine[:,k]
    C = testK - xApprox
    D = testGauss - xApprox
    E = testSOR -xApprox
    F = testIterative - xApprox
    G = testConjugatePre- xApprox
    H = testinfgradient - xApprox
    infJacob[:,k] = C[:,0]
    infGauss[:,k] = D[:,0]
    infSOR[:,k] = E[:,0]
    infIter[:,k] = F[:,0]
    infConjugate[:,k] = G[:,0]
    infgradient[:,k] = H[:,0]
    k = k+1
#a = np.array([[1,2],[3,4]]) 
# provides index
infJacob = np.absolute(infJacob)
infGauss = np.absolute(infGauss)
infSOR =  np.absolute(infSOR)    
infIter = np.absolute(infIter)
infConjugate = np.absolute(infConjugate)
infgradient = np.absolute(infgradient)
#provides index to print
normOfJacobi = np.argmax(infJacob, axis=0)
normGauss = np.argmax(infGauss, axis=0)
normSor = np.argmax(infSOR, axis=0)
normIter =np.argmax(infIter, axis=0)
normCon = np.argmax(infConjugate, axis=0)
normGradient = np.argmax(infgradient, axis =0)
# Stores 
Jacobix_xap= np.zeros((N))
normGaussx_xap = np.zeros((N))
normSorx_xap = np.zeros((N))
normIterx_xap = np.zeros((N))
normConx_xap = np.zeros((N))
normGradient_xap = np.zeros((N))
i = 0
for i in range(i,N):
    maxnumberOne = normOfJacobi[i]
    maxnumberTwo = normGauss[i]
    maxnumberThree = normSor[i]
    maxnumberFour = normIter[i]
    maxnumberFive= normCon[i]
    maxnumberSix = normGradient[i]
    #print("Jacobi number: \n ")
    #print(infJacob[maxnumberOne][i])
    Jacobix_xap[i] = infJacob[maxnumberOne][i]
    normGaussx_xap[i] =  infGauss[maxnumberTwo][i]
    normSorx_xap[i] = infSOR[maxnumberThree][i]
    normIterx_xap[i]= infIter[maxnumberFour][i]
    normConx_xap[i] = infConjugate[maxnumberFive][i]
    normGradient_xap[i] = infgradient[maxnumberSix][i]
print(" \n ----Points on Graph------ x-x* \n")
print(" \nJacobi \n")
print(Jacobix_xap ,"\n")
print("Gauss Seidel \n ")
print(normGaussx_xap, "\n")
print("SOR \n ")
print(normSorx_xap, "\n")
print("Iterative Refinement \n ")
print(normIterx_xap, "\n")
print("Preconditoned conjugate Gradient Method \n")
print(normConx_xap)
print("Conjugate Gradient Method ")
print(normGradient_xap)




























































    
    
    
    
plt.rcParams.update({'font.size': 10})	# set plot font size

# "these" are the numerical methods graphed
# x is the unique solution (3,4,-5)**t
#plt.plot(x, 'c-', linewidth=1,markersize=12)	
plt.plot(Jacobix_xap, 'r--', marker='v', linewidth=1,markersize=6)	
plt.plot(normGaussx_xap, 'p--', marker='s', linewidth=1,markersize=6)	
plt.plot(normSorx_xap, 'k--', marker='8', linewidth=1,markersize=6)	
plt.plot(normIterx_xap, 'y--', marker='H', linewidth=1,markersize=6)	
plt.plot(normConx_xap, 'm--', marker='p', linewidth=1,markersize=10)	
plt.plot(normGradient_xap, 'c--', marker='D', linewidth=1,markersize=6)	

    








plt.legend(['Jacobi','Gauss Seidel','SOR','IterativeRef','Preconjugate','Conjugate'], loc=0,fontsize =8)	# set legend and location


plt.xlabel('k')	# set x-axis label as t
plt.ylabel('||x- xApprox|| , N=8')	# set y-axis label as y(t)

    
    
plt.savefig('Example1.pdf', format='pdf', dpi=300)    
    

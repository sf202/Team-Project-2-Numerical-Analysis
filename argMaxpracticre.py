import numpy as np

aAbsolute = np.array([[4, 1,9], [2, 5,2],[1,2,4]], np.float) # left end point of interval [a,b]

test =np.zeros((3,3))

test[:,0] = aAbsolute[:,0]
absolutevalue = np.absolute(aAbsolute)

print(aAbsolute)
#0 1 2

#F = np.argmax(a)
#5
absolutevalueTest = np.argmax(aAbsolute, axis=0)
#array([0, 1, 2])
print (" \n This is for rows zero", absolutevalueTest ,"\n")

#Should print [1,1,2]
absolutevalueTestTwo = np.argmax(aAbsolute, axis=1)
#array([2, 2])
print ("This is for Columns One",absolutevalueTestTwo)

maxnumberOne = absolutevalueTest[0]
maxNumberTwo = absolutevalueTestTwo[0]

print(aAbsolute[0][maxnumberOne])


print(aAbsolute[0][maxNumberTwo])




#A = np.array([[1], [1,]], np.float) 
#b = np.array([[3],[0]], np.float)


#C = A-b

#print(C)












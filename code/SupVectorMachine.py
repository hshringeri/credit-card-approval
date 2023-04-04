import numpy as np
import cvxopt as co
import predict

def getY(column):
    y = np.zeros((len(column), 1))
    for i in range(len(column)):
        if (column[i] > 0.6):
            y[i] = 1
        else:
            y[i] = -1
        
    

    return y


def run(X,y):
    n = len(X) # rows
    d = np.shape(X)[1] # columns
    theta = np.zeros((d,1))
    H = np.eye(d)
    f = np.zeros(d)
    A = np.zeros((n,d))
    for i in range(n):
        for j in range(d):
            A[i,j] = -y[i]*X[i,j]
    b = np.full(n,-1)

   
    for i in range(n):
        if (predict.run(theta,X[i]) < 0):
            theta = np.array(co.solvers.qp(co.matrix(H,tc='d'),co.matrix(f,tc='d'),
            co.matrix(A,tc='d'),co.matrix(b,tc='d'))['x'])

    return theta

import numpy as np
import probclearn
import probcpredict
import math

def run(k,X,y):
    z = np.zeros((k,1))
    n = len(X) # rows
    d = np.shape(X)[1] # columns

    for i in range(0,k):
        T = set()
        S = set()

        start = math.floor(n * i/k)
        end =  math.floor(n * (i + 1)/k)

        for j in range(start, end):
            T.add(j)
        for b in range(0, n):
            S.add(b)
        S = S - T

        X_train = np.zeros((len(S),d))
        Y_train = np.zeros((len(S)))
        index = 0
        for s in S:
            X_train[index] = X[s]
            Y_train[index] = y[s]
            index = index + 1
        q, mu_positive, mu_negative, sigma2_positive, sigma2_negative = probclearn.run(X_train,Y_train)

        for t in T:
            if (y[t] != probcpredict.run(q, mu_positive, mu_negative, sigma2_positive, sigma2_negative, np.reshape(X[t], (d,1)))):   
                z[i] = z[i] + 1

        z[i] = z[i]/len(T)
        

    return z

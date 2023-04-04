import numpy as np
import numpy.linalg as la
# Input: numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: scalar q
# numpy vector mu_positive of d rows, 1 column
# numpy vector mu_negative of d rows, 1 column
# scalar sigma2_positive
# scalar sigma2_negative
def run(X,y):
    # Your code goes here
    n = len(X) # rows   
    d = np.shape(X)[1] # columns
    k_positive = 0
    k_negative = 0
    mu_positive = np.zeros((d,1))
    mu_negative = np.zeros((d,1))

    for t in range(0,n):
        if (y[t] == 1):
            k_positive = k_positive + 1
            for i in range(0,d):
                mu_positive[i]+=X[t][i]
            #mu_positive = np.add(mu_positive,X[t].T)
        else:
            k_negative = k_negative+ 1
            for i in range(0,d):
                mu_negative[i]+=X[t][i]
            #mu_negative = np.add(mu_negative,X[t].T)
    
    q = k_positive/n
    mu_positive = (1/k_positive)*(mu_positive)
    mu_negative = (1/k_negative)*(mu_negative)
    sigma2_positive = 0
    sigma2_negative = 0
    for t in range(0,n):
        Z = np.reshape(X[t], (d,1))
        if (y[t] == +1):
            #print(X[t])
            #print(np.reshape(X[t], (d,1)))
            #Z = np.reshape(X[t], (d,1))
            #for i in range(0,d):
                
            #    sigma2_positive = sigma2_positive + (la.norm(X[t][i]-mu_positive[i])) ** 2
            #    print("X[t][i]," , X[t][i])
            sigma2_positive = sigma2_positive + (la.norm(Z-mu_positive)) ** 2
        else:
            #for i in range(0,d):
                #("X[t][i]," , X[t][i])
                #sigma2_negative = sigma2_negative + (la.norm(X[t][i]-mu_negative[i]) * la.norm(X[t][i]-mu_negative[i]))
                #print("X[t][i]," , X[t][i])
            sigma2_negative = sigma2_negative + (la.norm(Z-mu_negative)) ** 2
       

    sigma2_positive = (1/(d*k_positive))*sigma2_positive
    sigma2_negative = (1/(d*k_negative))*sigma2_negative
    
    return q, mu_positive, mu_negative, sigma2_positive, sigma2_negative

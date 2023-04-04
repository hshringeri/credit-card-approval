import math
import numpy as np
import numpy.linalg as la
# Input: scalar q
# numpy vector mu_positive of d rows, 1 column
# numpy vector mu_negative of d rows, 1 column
# scalar sigma2_positive
# scalar sigma2_negative
# numpy vector z of d rows, 1 column
# Output: label (+1 or -1)
def run(q,mu_positive,mu_negative,sigma2_positive,sigma2_negative,z):
    # Your code goes here
    d = len(mu_positive) # rows
    one = math.log(q/(1-q))
    two = d/2 * math.log(sigma2_positive/sigma2_negative)
    three = (1/(2*sigma2_positive))*(la.norm(z-mu_positive) **2)
    four = (1/(2*sigma2_negative))*(la.norm(z-mu_negative) **2)
    #print(one - two - three + four)
    if ((one - two - three + four) > 0):
        label = 1
    else:
        label = -1
    
    return label

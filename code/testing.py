import numpy as np
import cvxopt as co
import predict
import csv
import pandas as pd
import SupVectorMachine

testingSet = pd.read_csv('data/test.csv')

# svm
def testSVM():
    testSVM = (testingSet).to_numpy()
    #print(trainSVM)
    testSVMy = (testingSet["AVERGE_STATUS"]).to_numpy()
    y = SupVectorMachine.getY(testSVMy)
    return SupVectorMachine.run(testSVM, y)

    
    

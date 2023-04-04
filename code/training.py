import numpy as np
import cvxopt as co
import predict
import csv
import pandas as pd
import SupVectorMachine

trainingSet = pd.read_csv('data/train.csv')

# svm
def trainSVM():
    trainSVM = (trainingSet).to_numpy()
    #print(trainSVM)
    trainSVMy = (trainingSet["AVERGE_STATUS"]).to_numpy()
    y = SupVectorMachine.getY(trainSVMy)
    return SupVectorMachine.run(trainSVM, y)

# k folds

def trainkFolds():
    trainSVM = (trainingSet).to_numpy()
    #print(trainSVM)
    trainSVMy = (trainingSet["AVERGE_STATUS"]).to_numpy()
    y = SupVectorMachine.getY(trainSVMy)
    return kFolds.run(50, trainSVM, y)

    








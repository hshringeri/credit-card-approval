import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm
import SupVectorMachine
import roc
import pandas as pd

testingSet = pd.read_csv('data/test.csv')
trainingSet = pd.read_csv('data/train.csv')

def run():
    roc.plotROC(testingSet, trainingSet)
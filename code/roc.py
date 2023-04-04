import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm
import SupVectorMachine



def plotROC(trainingSet, testingSet):
    trainY = (trainingSet["AVERGE_STATUS"]).to_numpy()
    testY = (testingSet["AVERGE_STATUS"]).to_numpy()
    train = (trainingSet).to_numpy()
    test = (testingSet).to_numpy()

    y = SupVectorMachine.getY(trainY)
    y2 = SupVectorMachine.getY(testY)

    clf = svm.SVC(random_state = 0)
    clf.fit(train, y)
    metrics.plot_roc_curve(clf, test, y2)
    plt.show()

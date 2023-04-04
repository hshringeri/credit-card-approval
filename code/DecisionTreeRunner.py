import decisionTree as DT
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


def status_to_class(df, train=True):
    df["BAD_CUSTOMER"] = 1
    df.loc[df["AVERGE_STATUS"] < 1, "BAD_CUSTOMER"] = 0
    if train:
        df.drop("AVERGE_STATUS", axis=1, inplace=True)
        train_cols = df.columns.to_list()
        train_cols.remove("BAD_CUSTOMER")
        return df, train_cols
    return df


def predict(tree, df_row):
    return DT.predict(tree, df_row)


def train(depth):
    train = pd.read_csv("../data/train.csv")
    predic_col = "BAD_CUSTOMER"
    train, train_cols = status_to_class(train)
    tree = DT.buildTree(train, train_cols, predic_col, max_depth=i)
    test = pd.read_csv("../data/test.csv")
    test = status_to_class(test, train=False)
    acc = DT.test_predictions(tree, test, "BAD_CUSTOMER")
    return tree, acc


accs = []
for i in range(int(sys.argv[1])):
    tree, acc = train(i)
    accs.append(acc)
    print(f"max_depth:{i}, Acc:{acc}")

plt.plot(accs)
plt.ylabel("Accuracy")
plt.xlabel("Tree Depths")
plt.title("Decision Tree Accuracy vs Depths Plot")
plt.show()

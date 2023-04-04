#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np


def BiClassEncode(df, col_name, val1, new_name=None):
    df.loc[df[col_name] == val1, col_name] = 1
    df.loc[df[col_name] != 1, col_name] = -1
    df[col_name] = df[col_name].astype(int)
    if new_name != None:
        df.rename(columns={col_name: new_name}, inplace=True)


def MultiClassEncode(df, col_name, dic, new_name=None):
    for key in dic.keys():
        df.loc[df[col_name] == key, col_name] = dic[key]
    df[col_name] = df[col_name].astype(int)
    if new_name != None:
        df.rename(columns={col_name: new_name}, inplace=True)


def Normalization(df, col_name, new_name):
    df[new_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std()
    df.drop(col_name, axis=1, inplace=True)


# Preprocess record


df_rec = pd.read_csv("../data/credit_record.csv")

df_rec.loc[df_rec["STATUS"] != "C", "STATUS"] = 1

df_rec.loc[df_rec["STATUS"] == "C", "STATUS"] = 0

df_rec.drop("MONTHS_BALANCE", axis=1, inplace=True)

df_rec["STATUS"] = df_rec["STATUS"].astype(int)

df_rec = df_rec.groupby(by="ID", dropna=True).mean()

df_rec.rename(columns={"STATUS": "AVERGE_STATUS"}, inplace=True)

# Merge

df_app = pd.read_csv("application_record.csv")
df = df_app.merge(df_rec, on="ID")

del df_rec
del df_app

# Biniary Categorical Encoding
BiClassEncode(df, "CODE_GENDER", "M", new_name="GENDER")

BiClassEncode(df, "FLAG_OWN_CAR", "Y", "OWN_CAR")

BiClassEncode(df, "FLAG_OWN_REALTY", "Y", "OWN_REALTY")

# Ordinal Encoding

edu_dic = {
    "Lower secondary": 0,
    "Secondary / secondary special": 1,
    "Incomplete higher": 2,
    "Higher education": 3,
    "Academic degree": 4,
}

MultiClassEncode(df, "NAME_EDUCATION_TYPE", edu_dic, "EDUCATION")

# Categorical Encoding

df.rename(columns={"NAME_FAMILY_STATUS": "MARITAL_STATUS"}, inplace=True)
one_hot = pd.get_dummies(df["MARITAL_STATUS"])
df.drop("MARITAL_STATUS", axis=1, inplace=True)
df = df.join(one_hot)


# Normalization

df["DAYS_BIRTH"] = df["DAYS_BIRTH"].abs()
Normalization(df, "DAYS_BIRTH", "DAYS_BIRTH_NORMALIZED")

df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].abs()
Normalization(df, "DAYS_EMPLOYED", "DAYS_EMPLOYED_NORMALIZED")

# Drop unuseable/useless attrubites

df.drop("OCCUPATION_TYPE", axis=1, inplace=True)
df.drop("NAME_INCOME_TYPE", axis=1, inplace=True)
df.drop("NAME_HOUSING_TYPE", axis=1, inplace=True)
df.drop("FLAG_MOBIL", axis=1, inplace=True)
df.drop("FLAG_WORK_PHONE", axis=1, inplace=True)
df.drop("FLAG_PHONE", axis=1, inplace=True)
df.drop("FLAG_EMAIL", axis=1, inplace=True)
df.drop("ID", axis=1, inplace=True)

df = df.sample(frac=1)
df.to_csv("preprocessed.csv")

train = df.head(500)
test = df.tail(500)

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

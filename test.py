import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

df = pd.read_csv("New Project 2_Ranking_2022-10-27_17-24-28.csv")

df = df.set_index("Name")
labels = [col for col in df.columns if "covid" in col.lower()]
tot_pop = [col for col in df.columns if "total" in col.lower()][0]
y = df[labels]
# Normalized y
y_lab = y[labels[11]]/df[tot_pop]
# df = df[df[labels[11]].isna() == False]
df = df[y_lab.isna() == False]
y_lab = y_lab[y_lab.isna() == False]
x = df.drop(labels, axis=1)
x = df.drop(tot_pop, axis=1)
bad_col = [c for c in x.columns if x[c].isna().sum() > 10]
x = x.drop(bad_col, axis=1)
print(y_lab.shape)
print(x.shape)

y_lab = np.where(y_lab > np.median(y_lab), 1, 0)

for c in x.columns:
    x[c].fillna(value=x[c].mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(x, y_lab, test_size=0.15, random_state=42)

# for c in X_train.columns:
#     if X_train[c].isna().sum()> 1:
#         print(c)
# X_train.isna().sum().sum()
# lm = LinearRegression().fit(X_train, y_train)

for c in X_train.columns:
    if X_train[c].isna().sum()> 1:
        print(c)
print(X_train.isna().sum().sum())
clf = LogisticRegression(penalty="l1", solver="liblinear").fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

y_train_pred = clf.predict(X_train)

X_train_low = X_train[y_train_pred == 0]
X_train_high = X_train[y_train_pred == 1]

y_train_low = y_train[y_train_pred == 0]
y_train_high = y_train[y_train_pred == 1]

X_test_low = X_test[y_pred == 0]
X_test_high = X_test[y_pred == 1]

y_test_low = y_test[y_pred == 0]
y_test_high = y_test[y_pred == 1]

lm_low = LinearRegression().fit(X_train_low, y_train_low)
lm_high = LinearRegression().fit(X_train_high, y_train_high)

y_pred_low = lm_low.predict(X_test_low)
y_pred_high = lm_low.predict(X_test_high)


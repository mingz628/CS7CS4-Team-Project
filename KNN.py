import csv
import os

import numpy as np
import pandas as pd
import row as row
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("group.csv", parse_dates=True)  # read file
# X = np.column_stack((x, y))
co = df.loc[df['Specie'] == 'co']
co = co.iloc[:, 7]
no2 = df.loc[df['Specie'] == 'no2']
no2 = no2.iloc[:, 7]
o3 = df.loc[df['Specie'] == 'o3']
o3 = o3.iloc[:, 7]
pm10 = df.loc[df['Specie'] == 'pm10']
pm10 = pm10.iloc[:, 7]
pm25 = df.loc[df['Specie'] == 'pm25']
pm25 = pm25.iloc[:, 7]
so2 = df.loc[df['Specie'] == 'so2']
so2 = so2.iloc[:, 7]
temperature = df.loc[df['Specie'] == 'temperature']
temperature = temperature.iloc[:, 7]
pressure = df.loc[df['Specie'] == 'pressure']
pressure = pressure.iloc[:, 7]
humidity = df.loc[df['Specie'] == 'humidity']
humidity = humidity.iloc[:, 7]
windspeed = df.loc[df['Specie'] == 'windspeed']
windspeed = windspeed.iloc[:, 7]
dew = df.loc[df['Specie'] == 'dew']
dew = dew.iloc[:, 7]
X = np.column_stack((windspeed, humidity, pressure, temperature))
X2 = np.column_stack((humidity, temperature,windspeed))
X3 = np.column_stack((windspeed, pressure))
IAQI_CO = df.loc[df['Specie'] == 'co']
IAQI_CO = IAQI_CO.iloc[:, 9]
IAQI_NO2 = df.loc[df['Specie'] == 'no2']
IAQI_NO2 = IAQI_NO2.iloc[:, 11]
IAQI_O3 = df.loc[df['Specie'] == 'o3']
IAQI_O3 = IAQI_O3.iloc[:, 12]
IAQI_PM10 = df.loc[df['Specie'] == 'pm10']
IAQI_PM10 = IAQI_PM10.iloc[:, 14]
IAQI_PM25 = df.loc[df['Specie'] == 'pm25']
IAQI_PM25 = IAQI_PM25.iloc[:, 13]
IAQI_SO2 = df.loc[df['Specie'] == 'so2']
IAQI_SO2 = IAQI_SO2.iloc[:, 10]
# print(IAQI_PM25)
U = np.column_stack((IAQI_CO, IAQI_NO2, IAQI_O3, IAQI_PM25))
# print(X)
# print(temperature)
x_train, x_test, y_train, y_test = train_test_split(X, co, test_size=0.3)
# print(U)
# Umax = U.max(axis=1)
# df['AQI'] = U.max(axis=1)
# print(Umax)
# newdf = pd.DataFrame()
# newdf['AQI'] = U.max(axis=1)
# print(newdf)
#
# header = ['AQI']
# data=newdf
# path ="AQI.csv"
# df = pd.DataFrame(data=newdf)
# if not os.path.exists(path):
#     df.to_csv(path, header=['AQI'], index=False, mode='a')
# else:
#     df.to_csv(path, header=False, index=False, mode='a')
AQIdf = pd.read_csv("AQI.csv")  # read file
z = AQIdf.iloc[:, 1]
# print(z)
def a():
    k_range = range(1, 30)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X2, z, cv=5, scoring='neg_log_loss')
        print(k)
        print(scores.mean())
        k_scores.append(scores.mean())
    print('max')
    print(max(k_scores))


def b():
    # knn = KNeighborsClassifier(n_neighbors=8)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X2, z)
    train_predict2 = knn.predict(X2)
    print(confusion_matrix(y_true=z, y_pred=train_predict2, labels=[1,2,3,4,5]))


def c():
    # knn = KNeighborsClassifier(n_neighbors=8)
    knn = KNeighborsClassifier(n_neighbors=27)
    knn.fit(X2, z)
    train_predict2 = knn.predict(X2)
    score = knn.predict_proba(X2)
    fpr, tpr, thresholds = roc_curve(y_true=z, y_score=score[:, 1], pos_label=4)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], lw=2, ls="--", label="random")
    plt.plot([0, 0, 1], [0, 1, 1], lw=2, ls="-.", label="perfect")
    plt.xlabel('False Positive Rate(FPR)', fontsize=16)
    plt.ylabel('True Positive Rate(TPR)', fontsize=16)
    plt.grid()
    plt.title(f"ROC{auc(fpr, tpr):.3f}", fontsize=16)
    plt.legend()
    plt.show()


def d():
    plt.rc('font', size=18)
    # knn = KNeighborsClassifier(n_neighbors=8)
    knn = KNeighborsClassifier(n_neighbors=27)
    knn.fit(X2, z)
    train_predict2 = knn.predict(X2)
    score = knn.predict_proba(X2)
    fpr, tpr, thresholds = roc_curve(y_true=z, y_score=score[:, 1], pos_label=3)
    plt.plot(fpr, tpr, label='knn')
    Xtrain, Xtest, ytrain, ytest = train_test_split(X2, z, test_size=0.2)
    dummy = DummyClassifier(strategy="uniform").fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest, dummy.predict_proba(Xtest)[:, 1],pos_label=3)
    plt.plot(fpr, tpr, c='y', label='baseline', linestyle='--')
    plt.xlabel('False Positive Rate(FPR)', fontsize=16)
    plt.ylabel('True Positive Rate(TPR)', fontsize=16)
    plt.legend()
    plt.show()
if __name__=='__main__':
    # a()
    # b()
    c()

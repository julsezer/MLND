# Machine Learning Nano Degree Program - Capstone Project
# Trading algorithm for THYAO stock listed in IMKB (Istanbul)
# Target value will be moving average(MA) of next 14 days

# PART 1 - PREPARING THE DATASET

#import libraries. Talib is a library for technical analysis functions
import numpy as np
import pandas as pd
import talib

# Import time series data of THYAO from July 2013 to May 2018
data = pd.read_csv("THYAO.IS.csv", sep=",")

# Drop indexes that have no value
data = data.dropna(axis=0)

# Drop date & Adj Close column
data = data.drop(["Date"], axis=1)
data = data.drop(["Adj Close"], axis=1)

# Calculate Absolute Price Oscillator
data["APO"] = talib.APO(data["Close"], fastperiod=12, slowperiod=26, matype=0)

# Calculate Aroon Oscillator
data["AROONOSC"] = talib.AROONOSC(data["High"], data["Low"], timeperiod=14)

# Calculate MACD
data["MACD"], data["MACD Sig"], data["MACD Hist"] = talib.MACD(data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)

# Calculate Momentum
data["Momentum"] = talib.MOM(data["Close"], timeperiod=10)

# Calculate RSI
data["RSI"] = talib.RSI(data["Close"], timeperiod=14)

# Calculate Stochastic
data["SLowk"], data["Slowd"] = talib.STOCH(data["High"], data["Low"], data["Close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

# Calculate William' %R
data["Williams"] = talib.WILLR(data["High"], data["Low"], data["Close"], timeperiod=14)

# Calculate Bollinger Bands
data["UpBand"], data["MidBand"], data["LowBand"] = talib.BBANDS(data["Close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)

# Calculate Parabolic SAR
data["P-SAR"] = talib.SAR(data["High"], data["Low"], acceleration=0, maximum=0)

# Calculate Weighted Moving Average
data["WMA"] = talib.WMA(data["Close"], timeperiod=30)

# Calculate Chaikin A/D Oscillator
data["Chaikin"] = talib.ADOSC(data["High"], data["Low"], data["Close"], data["Volume"], fastperiod=3, slowperiod=10)

# Calculate Moving Average 28
data["MA28"] = talib.MA(data["Close"], timeperiod=28, matype=0)

# Calculate Hilbert Transform - Trend vs Cycle Mode
data["Hilbert"] = talib.HT_TRENDMODE(data["Close"])


""" Calculate the target variable. If the Closing price is higher than 3%
of any upfront closing price in 10 days, set the target variable as 1. Else 0.
Aim is to balance the ratio of True and False in order to avoid imbalanced class problem """

data = data.reset_index(drop=True)

r, c = data.shape
count = r - 10
data["Target"] = 0
for i in range(count):
    for n in range(1,10):
        if (data.loc[i+n-1, "Close"] > (data.loc[i, "Open"] * 1.03)):
            data.loc[i, "Target"] = 1
            break


# Drop NaN rows and reset the index
data = data.dropna(axis=0)
data = data.reset_index(drop=True)

data.to_csv("data.csv", sep=",", decimal=',', float_format='%.3f')


# Split the dataset into training and testing sets. The ratio is 80/20
train_end = int(np.floor(r*0.8))
X_train = data.iloc[0:train_end, 3:c]
y_train = data.iloc[0:train_end, c]
X_test = data.iloc[train_end:r, 3:c]
y_test = data.iloc[train_end:r, c]
#y_test = y_test.reset_index(drop=True)

# Preprocess and standardize the data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
components = pca.components_

## PART 2 - Supervised Learning

## KNN
#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)

## Random Forrest Classifier (RFC)
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators = 200, criterion = 'gini', max_depth = None)
#clf.fit(X_train, y_train)

## Support Vector Classifier (SVC)
#from sklearn.svm import SVC
#clf = SVC(kernel = 'rbf', C=1, gamma=0.2)
#clf.fit(X_train, y_train)

# Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, algorithm="SAMME.R")
clf.fit(X_train, y_train)


# PART 3 - PREDICTING THE RESULT AND APPLYING TRADING STRATEGY
y_pred = clf.predict(X_test)


from sklearn.metrics import confusion_matrix, fbeta_score
cm = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_train)
fbeta_score = fbeta_score(y_test, y_pred, beta=0.5)


## Grid Search
## Applying Grid Search to find the best model and the best parameters
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import fbeta_score, make_scorer
#fbeta_scorer = make_scorer(fbeta_score, beta=0.5)
#clf = AdaBoostClassifier()
#parameters = [{'n_estimators': [150,200,250,300,350,400,450,500,550,600,650,700], 
#               'learning_rate': [0.01,0.02,0.03,0.04,0.05,0.06],
#               "algorithm": ["SAMME.R", "SAMME"],
#               }]
#grid_search = GridSearchCV(estimator = clf,
#                           param_grid = parameters,
#                           scoring = fbeta_scorer,
#                           cv = 10,
#                           n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#best_score = grid_search.best_score_
#best_parameters = grid_search.best_params_


# Algorithm
profit = 0
cost = 0
budget = 8.68
has_stock = False
data_trade = data.iloc[train_end:r, 0:c]
data_trade["Target"] = y_pred.astype(int)
data_trade = data_trade.reset_index(drop=True)
k, n = data_trade.shape
count=0

# Time series of total asset value for comparison graph
values = []
agent_asset = pd.DataFrame({"Total" : []})

for i in range(k):
    # For time series value graph
    if not has_stock:
        values.append(budget)
    if has_stock:
        values.append(budget+cost)
        
    # Trading algorithm   
    if(has_stock==False and (data_trade.loc[i, "Target"] == 1)):
        cost = data_trade.loc[i, "Open"]
        budget -= cost
        print (i, ": cost:", cost, ", budget: ", budget, ", profit:", profit, ", buy")
        has_stock = True
        count += 1
    elif (has_stock == True and ((data_trade.loc[i, "Open"]) > (cost * 1.03))):
        profit += (data_trade.loc[i, "Open"] - cost)
        budget += data_trade.loc[i, "Open"]
        print (i, ":  budget:", budget, ", profit:", profit, ", sell")
        has_stock = False
        count += 1
    # Stop-loss
    elif (has_stock == True and ((data_trade.loc[i, "Open"]) < (cost / 1.03))):
        profit += (data_trade.loc[i, "Open"] - cost)
        budget += data_trade.loc[i, "Open"]
        print (i, ":  budget:", budget, ", profit:", profit, ", sell")
        has_stock = False
        count += 1

agent_asset["Total"] = values
agent_asset["Total"].to_csv("asset.csv", sep=",", decimal=',', float_format='%.3f')
data_trade["Open"].to_csv("natural.csv", sep=",", decimal=',', float_format='%.3f')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.utils import to_categorical
from keras import optimizers
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from matplotlib.pyplot import figure, show
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection._split import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold



df = pd.read_csv('exp31.csv')
# Remove NaN values from dataframe
df.fillna(0, inplace=True)
# Remove 0 values from fun column
df = df[df.fun != 0]
X = df.iloc[:, [8,9,10,13,14,15,16,19,20,21,22,25,26,27,28,31,32,33,34,37,38,39,40,43,44,45,46,49,50,51,52,55,56,57,58,61,62,63,64]].values
Y = df.iloc[:, 3].values

X_new = SelectKBest(f_regression, k=20).fit_transform(X, Y)
print("**********")
print(X_new[0])
print(X_new[10])

# Scale input values to a range (-1, 1)
scaler = MinMaxScaler((-1, 1), True)
X_new = scaler.fit_transform(X_new)

X_new_train, X_new_test, Y_train, Y_test = train_test_split(X_new, Y, test_size=0.20, random_state=42)

seed = 7
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

for train, test in kfold.split(X_new, Y):
    model = Sequential()
    model.add(Dense(20, input_dim=X_new.shape[1], activation='linear'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    model.fit(X_new[train], Y[train], verbose=2, epochs=150)
    scores = model.evaluate(X_new[test], Y[test], verbose=0)
    print("%s: %f" % (model.metrics_names[1], scores[1]))
    print("RMSE: %f" % (np.sqrt(scores[1])))
    cvscores.append(scores[1] * 100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
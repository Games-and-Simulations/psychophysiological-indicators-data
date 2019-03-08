import numpy as np
np.random.seed(7)
import random as rn
rn.seed(7)
import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection._split import train_test_split
from keras.callbacks import EarlyStopping
from keras import backend as K
from tensorflow import set_random_seed
set_random_seed(7)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
import eli5
from eli5.sklearn import PermutationImportance
# Set every random seed to get replicable results

# Read CSV file, remove NaN and 0 values from dataframe and copy values to numpy array
df = pd.read_csv('exp31.csv')
df.fillna(0, inplace=True)
df = df[df.fun != 0]
X = df.iloc[:, [8,9,10,13,14,15,16,19,20,21,22,25,26,27,28,31,32,33,34,37,38,39,40,43,44,45,46,49,50,51,52,55,56,57,58,61,62,63,64]].values
Y = df.iloc[:, 3].values

# Select k best features for model inputs
X = SelectKBest(f_regression, k=20).fit_transform(X, Y)
# Scale input values to a range (-1, 1)
scaler = MinMaxScaler((-1, 1), True)
X = scaler.fit_transform(X)
# Shuffled (default) split into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=7)
# Create model
model = Sequential()
model.add(Dense(128, input_dim=X.shape[1], activation='linear'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

# Define rmse as a metric
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
adam = optimizers.Adam(lr=0.001)

model.compile(loss='mean_squared_error', optimizer=adam, metrics=[rmse])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode="auto")
history = model.fit(X, Y, validation_data=(X_test, Y_test), verbose=2, epochs=5)
pred = model.predict(X_test)

# Calculate and print scores
score = np.sqrt(metrics.mean_squared_error(pred, Y_test))
score3 = metrics.mean_squared_error(pred, Y_test)
score2 = np.sqrt(metrics.mean_absolute_error(pred, Y_test))
print("Score (RMSE): {}".format(score))
print("Score (MSE): {}".format(score3))
print("Score (MAE): {}".format(score2))

'''
# Calculate and show feature importance scores
perm = PermutationImportance(model, scoring='neg_mean_squared_error', random_state=7).fit(X,Y)
eli5.show_weights(perm, feature_names = X.columns.tolist())
'''

# Summarize history and show plots
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['rmse'])
plt.title('metrics')
plt.ylabel('loss, rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test', 'rmse'], loc='upper right')
plt.show()

'''
# 
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

#print(X_new.shape)
for train, test in kfold.split(X, Y):
    model = Sequential()
    model.add(Dense(128, input_dim=X.shape[1], activation='linear'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=[rmse])
    history = model.fit(X, Y, validation_data=(X_test, Y_test), verbose=2, epochs=5)
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.3f%%" % (model.metrics_names[1], scores[1]))
    print("RMSE: %.3f%%" % (np.sqrt(scores[1])))
    cvscores.append(scores[1] * 100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

'''

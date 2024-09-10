import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

df = pd.DataFrame(np.random.randint(0, 255, size=(1000, 2)), columns=list('AB'))
# print(df)
df['Q'] = df.sum(axis=1)  # unsigned
df['R'] = df.Q.apply(lambda x: format(int(x), '09b'))
#df['Q']=df['A']*df['B']
#df['R']=df.Q.apply(lambda x: format(int(x), '016b'))
print('Before bitflip\n', df)

# flipping nth bits
df.R.iloc[501:1000] = df.R.iloc[501:1000].str.slice(stop=0) + (1 - df.R.iloc[501:1000].str.get(0).astype(int)).astype(str) + df.R.iloc[501:1000].str.slice(start=1)
df['T'] = df.R.apply(lambda x: str(int(x, 2)))
print('After nth bitflip in R\n', df)

# final dataframe
X = df.drop(['Q', 'R'], axis=1)
#print(X)
y = np.repeat([0, 1], [500, 500])
# print(y)

# test_train_split
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.20)
# print('X_training set\n', X1)
# print('y_training label\n', y1)
print('X_testing set\n', X2)
#print('y_testing label\n', y2)

# SVC algorithm
svc1 = SVC(kernel='linear', C=1, degree=1, gamma='auto')
svc1.fit(X1, y1)
y_pred1 = svc1.predict(X2)
TN1, FP1, FN1, TP1 = confusion_matrix(y2, y_pred1, labels=[0, 1]).ravel()
accuracy1 = (TP1 + TN1) / (TP1 + FP1 + TN1 + FN1) * 100
print('accuracy1', accuracy1)


# SVC algorithm
svc3 = SVC(kernel='linear', C=2, degree=1, gamma='auto')
svc3.fit(X1, y1)
y_pred3 = svc3.predict(X2)
TN3, FP3, FN3, TP3 = confusion_matrix(y2, y_pred3, labels=[0, 1]).ravel()
accuracy3 = (TP3 + TN3) / (TP3 + FP3 + TN3 + FN3) * 100
print('accuracy3', accuracy3)

#SVC algorithm
svc5 = SVC(kernel='linear', C=3, degree=1, gamma='auto')
svc5.fit(X1, y1)
y_pred5 = svc5.predict(X2)
TN5, FP5, FN5, TP5 = confusion_matrix(y2, y_pred5, labels=[0, 1]).ravel()
accuracy5 = (TP5 + TN5) / (TP5 + FP5 + TN5 + FN5) * 100
print('accuracy5', accuracy5)

#SVC algorithm
svc7 = SVC(kernel='linear', C=4, degree=1, gamma='auto')
svc7.fit(X1, y1)
y_pred7 = svc7.predict(X2)
TN7, FP7, FN7, TP7 = confusion_matrix(y2, y_pred7, labels=[0, 1]).ravel()
accuracy7 = (TP7 + TN7) / (TP7 + FP7 + TN7 + FN7) * 100
print('accuracy7', accuracy7)

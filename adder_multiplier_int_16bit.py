import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randint(0,65536, dtype='uint32', size=(10,2)), columns=list('AB'))
#df['Q']=df['A'] + df['B']
#df['R']=df.Q.apply(lambda x: format(int(x), '017b'))
df['Q']=df['A'] * df['B']
df['R']=df.Q.apply(lambda x: format(int(x), '032b'))
print('Before bitflip\n',df)

#flipping bits
j = df.columns.get_loc("R")
df.iloc[5:10, j] = df.iloc[5:10, j].str.slice(stop=0) + (1 - df.iloc[5:10, j].str.get(0).astype(int)).astype(str) + df.iloc[5:10, j].str.slice(start=1)
#df.iloc[1001:2000, j] = df.iloc[1001:2000, j].str.slice(stop=0) + (1 - df.iloc[1001:2000, j].str.get(0).astype(int)).astype(str) + df.iloc[1001:2000, j].str.slice(start=1)
df['T']=df.R.apply(lambda x: str(int(x,2)))
print('After nth bitflip in R\n',df)

#final dataframe
X=df.drop(['Q', 'R'], axis=1)
print(X)
y = np.repeat([0, 1], [5,5])
print(y)

#data preparation
from sklearn.model_selection import train_test_split
X1, X2, y1, y2 = train_test_split(X, y, test_size = 0.20, random_state = 0)
#print('X_training set\n', X1)
#print('y_training label\n', y1)
print('X_testing set\n', X2)
print('y_testing label\n', y2)

#SVC algorithm
from sklearn.svm import SVC
svc = SVC(kernel='linear', C=1, degree=2, gamma=0.00001)
svc.fit(X1, y1)
y_pred = svc.predict(X2)
print('y_predicted label\n', y_pred)

#accuracy calculation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y2, y_pred)
TN, FP, FN, TP = confusion_matrix(y2, y_pred, labels=[0,1]).ravel()
accuracy = (TP + TN) / (TP + FP + TN + FN)*100
print('accuracy', accuracy)
#print('Error', 1 - accuracy)

import numpy as np
import pandas as pd

def truthTable(inputs=8):
    if inputs == 8:
        table = []
        for a3 in range(0, 2):
            for a2 in range(0, 2):
                for a1 in range(0, 2):
                    for a0 in range(0, 2):
                        for b3 in range(0, 2):
                            for b2 in range(0, 2):
                                for b1 in range(0, 2):
                                    for b0 in range(0, 2):
                                        table.append([a3, a2, a1, a0, b3, b2, b1, b0])
    return table

df = pd.DataFrame(truthTable(), columns=["A", "B", "C", "D", "E", "F", "G", "H"])
df['A1'] = pd.Series(df.fillna('').values.tolist()).map(lambda x: ''.join(map(str, x)))  # Concatenate all columns in a pandas dataframe
df['A11'] = df.A1.apply(lambda x: (int(x, 2)))
df['A22'] = np.random.permutation(df.A11) #random position of values
#df['A2'] = df.A22.apply(lambda x: format(int(x), '08b'))
df['B22'] = np.random.permutation(df.A11)
#df['B2'] = df.B22.apply(lambda x: format(int(x), '08b'))
df['C22'] = df['A22'] + df['B22']
df['C11'] = df.C22.apply(lambda x: format(int(x), '09b'))
#df['C22'] = df['A22'] * df['B22']
#df['C11'] = df.C22.apply(lambda x: format(int(x), '016b'))
print('Before bitflip\n',df)

# flipping bits
# df.C1.iloc[129:256]= df.C1.iloc[129:256].str.slice(stop=-1) + (1 - df.C1.iloc[129:256].str.slice(start=-1).astype(int)).astype(str)
j = df.columns.get_loc('C11')
df.iloc[129:256, j] = df.iloc[129:256, j].str.slice(stop=0) + (1 - df.iloc[129:256, j].str.get(0).astype(int)).astype(str) + df.iloc[129:256, j].str.slice(start=1)
df['C1'] = df.C11.apply(lambda x: (int(x, 2)))
print('After nth bitflip in C11\n',df)

# final dataframe
X = df[['A22', 'B22', 'C1']]
#X = df[['A2', 'B2', 'C11']]
print(type(X))
print(X)
y = np.repeat([0, 1], [128, 128])
print(y)

# data preparation
from sklearn.model_selection import train_test_split
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.10, random_state=0)
print('X_training set\n', X1)
print('y_training label\n', y1)
# print('X_testing set\n', X2)
print('y_testing label\n', y2)

# SVC algorithm
from sklearn.svm import SVC
svc = SVC(kernel='linear', C=2, degree=1, gamma='auto')
svc.fit(X1, y1)
y_pred = svc.predict(X2)
print('y_predicted label\n', y_pred)

# accuracy calculation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y2, y_pred)
TN, FP, FN, TP = confusion_matrix(y2, y_pred, labels=[0, 1]).ravel()
accuracy = (TP + TN) / (TP + FP + TN + FN)*100
print('accuracy', accuracy)
#print('Error', 100 - accuracy)

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
num_simulations = 1
simulation_results1 = []
simulation_results3 = []
simulation_results5 = []
simulation_results7 = []

df = pd.DataFrame(np.random.randint(0, 255, size=(1000, 2)), columns=list('AB'))
# print(df)
#df['Q'] = df.sum(axis=1)  # unsigned
#df['R'] = df.Q.apply(lambda x: format(int(x), '09b'))
df['Q']=df['A']*df['B']
df['R']=df.Q.apply(lambda x: format(int(x), '016b'))
print('Before bitflip\n', df)

# flipping bits
#j = df.columns.get_loc("R")
#df.iloc[501:1000, j] = df.iloc[501:1000, j].str.slice(stop=0) + (1 - df.iloc[501:1000, j].str.get(1).astype(int)).astype(str) + df.iloc[501:1000, j].str.slice(start=1)
df.R.iloc[501:1000] = df.R.iloc[501:1000].str.slice(stop=0) + (1 - df.R.iloc[501:1000].str.get(0).astype(int)).astype(str) + df.R.iloc[501:1000].str.slice(start=1)
df['T'] = df.R.apply(lambda x: str(int(x, 2))).astype(int)
#df=df.astype(int)
print('After nth bitflip in R\n', df)
"""
def flip_last_n_bits(binary_string):  # change here
    return binary_string[:-4] + ''.join(['1' if bit == '0' else '0' for bit in binary_string[-4:]])
# Function to flip the first two bits of a string
def flip_first_n_bits(s):
    if len(s) < 2:
        return s
    else:
        return ''.join(['1' if bit == '0' else '0' for bit in s[:4]]) + s[4:]
# Apply the function to each row in the DataFrame, #change here
df.loc[:500, 'R'] = df.loc[:500, 'R'].apply(flip_first_n_bits)
df['T'] = df.R.apply(lambda x: str(int(x, 2))).astype(int)
print('After nth bitflip in R\n', df)
"""
def min_max_normalize(column):
    min_val = column.min()
    max_val = column.max()
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column
# final dataframe
X = df.drop(['Q', 'R'], axis=1).apply(min_max_normalize)
print(X)
y = np.repeat([0, 1], [500, 500])
# print(y)

for _ in range(num_simulations):
    # data preparation
    from sklearn.model_selection import train_test_split
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.20)
    # print('X_training set\n', X1)
    # print('y_training label\n', y1)
    print('X_testing set\n', X2)
    #print('y_testing label\n', y2)

    # SVC algorithm
    svc1 = SVC(kernel='linear', C=1, degree=1, gamma='auto')
    svc1.fit(X1, y1)
    y_pred1 = svc1.predict(X2)
    TN, FP, FN, TP = confusion_matrix(y2, y_pred1, labels=[0, 1]).ravel()
    accuracy1 = (TP + TN) / (TP + FP + TN + FN) * 100
    print('accuracy1', accuracy1)
    simulation_results1.append(accuracy1)
    # SVC algorithm
    svc3 = SVC(kernel='linear', C=3, degree=1, gamma='auto')
    svc3.fit(X1, y1)
    y_pred3 = svc3.predict(X2)
    TN3, FP3, FN3, TP3 = confusion_matrix(y2, y_pred3, labels=[0, 1]).ravel()
    accuracy3 = (TP3 + TN3) / (TP3 + FP3 + TN3 + FN3) * 100
    print('accuracy3', accuracy3)
    simulation_results3.append(accuracy3)
    # SVC algorithm
    svc5 = SVC(kernel='linear', C=6, degree=1, gamma='auto')
    svc5.fit(X1, y1)
    y_pred5 = svc5.predict(X2)
    TN5, FP5, FN5, TP5 = confusion_matrix(y2, y_pred5, labels=[0, 1]).ravel()
    accuracy5 = (TP5 + TN5) / (TP5 + FP5 + TN5 + FN5) * 100
    print('accuracy5', accuracy5)
    simulation_results5.append(accuracy5)
    # SVC algorithm
    svc7 = SVC(kernel='linear', C=9, degree=1, gamma='auto')
    svc7.fit(X1, y1)
    y_pred7 = svc7.predict(X2)
    TN7, FP7, FN7, TP7 = confusion_matrix(y2, y_pred7, labels=[0, 1]).ravel()
    accuracy7 = (TP7 + TN7) / (TP7 + FP7 + TN7 + FN7) * 100
    print('accuracy7', accuracy7)
    simulation_results7.append(accuracy7)

mean1 = sum(simulation_results1) / num_simulations
mean3 = sum(simulation_results3) / num_simulations
mean5 = sum(simulation_results5) / num_simulations
mean7 = sum(simulation_results7) / num_simulations
print("Simulation resultsC1:", simulation_results1)
print("Simulation resultsC3:", simulation_results3)
print("Simulation resultsC5:", simulation_results5)
print("Simulation resultsC7:", simulation_results7)
print("MeanC1:", mean1)
print("MeanC3:", mean3)
print("MeanC5:", mean5)
print("MeanC7:", mean7)

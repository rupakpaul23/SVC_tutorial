import pandas as pd
import numpy as np

df = pd.read_csv  (r'folderpath_csvfilev')
#data.columns=['C']
df.columns = ['enc', '16-QAM']
data2 = df.drop('enc', axis=1).dropna()

#drop duplicate values
data2=data2.drop_duplicates()
data2 = data2.sort_values('16-QAM').reset_index(drop=True)
data2['16-QAM'] = data2['16-QAM'].str.replace('i', 'j').apply(complex)
print('qam_w/o duplicate\n',data2)

# Create a DataFrame with 16-QAM points and random integer values
data1 = pd.DataFrame({'Integer': np.random.randint(1, 10, 16)})
data=data1.join(data2['16-QAM'])
print('Int + 16-QAM\n',data)
pd.set_option('display.max_columns', None)

# Create a new DataFrame with four columns for the four quadrants
data['Real'] = data['16-QAM'].apply(lambda x: x.real)
data['Imaginary'] = data['16-QAM'].apply(lambda x: x.imag)
data['Quadrant'] = ''
data.loc[(data['Real'] >= 0) & (data['Imaginary'] >= 0), 'Quadrant'] = 'I'
data.loc[(data['Real'] < 0) & (data['Imaginary'] >= 0), 'Quadrant'] = 'II'
data.loc[(data['Real'] < 0) & (data['Imaginary'] < 0), 'Quadrant'] = 'III'
data.loc[(data['Real'] >= 0) & (data['Imaginary'] < 0), 'Quadrant'] = 'IV'

# Separate DataFrames for each quadrant and their corresponding integer values
quadrant_I = data[data['Quadrant'] == 'I'].drop(columns=['Quadrant'])
quadrant_II = data[data['Quadrant'] == 'II'].drop(columns=['Quadrant'])
quadrant_III = data[data['Quadrant'] == 'III'].drop(columns=['Quadrant'])
quadrant_IV = data[data['Quadrant'] == 'IV'].drop(columns=['Quadrant'])

# Reset the indices for each quadrant DataFrame
quadrant_I.reset_index(drop=True, inplace=True)
quadrant_II.reset_index(drop=True, inplace=True)
quadrant_III.reset_index(drop=True, inplace=True)
quadrant_IV.reset_index(drop=True, inplace=True)

# Concatenate the DataFrames for all four quadrants and their integer values
all_quadrants = pd.concat([quadrant_I, quadrant_II, quadrant_III, quadrant_IV], axis=1)
all_quadrants=all_quadrants.drop('16-QAM', axis=1)
# Display the combined DataFrame
print(all_quadrants)

####################################################
from random import uniform, seed
import matplotlib.pyplot as plt
seed(10)
dframe4=all_quadrants.iloc[:,10:12]
print(dframe4)
# randomly around their original values; (n=20%,30%,40%)
randomized_data = {}
for variable, value_lst in dframe4.items():
    new_lst = []
    for value in value_lst:
        new_random_value = uniform(0.4, 1.8) * value
        new_lst.append(new_random_value)
    randomized_data[variable] = new_lst
# to clarify answer in output graph
for i in range(len(dframe4["Real"])):
    points_x_vals = dframe4["Real"][i], randomized_data["Real"][i]
    points_y_vals = dframe4["Imaginary"][i], randomized_data["Imaginary"][i]
    plt.plot(points_x_vals, points_y_vals, "black")
# actual plotting of data

plt.scatter(dframe4["Real"], dframe4["Imaginary"], label="original data")
plt.scatter(randomized_data["Real"], randomized_data["Imaginary"], label="\"randomized\" data")
plt.legend()
plt.grid()
plt.show()

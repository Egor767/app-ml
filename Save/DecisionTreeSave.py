import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import pickle

data = pd.read_csv("Data/csgo_taskv2.csv")

data_major_true = data[data['bomb_planted'] == True]
data_major_false = data[data['bomb_planted'] == False]

data_major_false_undersampled = data_major_false.sample(len(data_major_true))
data1 = pd.concat([data_major_false_undersampled, data_major_true], axis = 0)

y_cs = data1['bomb_planted']

data1 = data1.drop(['map'], axis=1)
data1 = data1.drop(['Unnamed: 0'], axis=1)
x_cs = data1.drop(['bomb_planted'], axis=1)
x_train_cs, x_test_cs, y_train_cs, y_test_cs = train_test_split(x_cs, y_cs, test_size=0.3)


tree_cl = DecisionTreeClassifier(criterion = 'gini', max_depth = 6, min_samples_split = 4)
tree_cl.fit(x_train_cs, y_train_cs)

with open('DecisionTreeClassifier.pkl','wb') as file:
    pickle.dump(tree_cl,file)


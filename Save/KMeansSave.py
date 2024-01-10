import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
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
x_train, x_test, y_train, y_test = train_test_split(x_cs, y_cs, test_size=0.3)

#{'algorithm': 'elkan', 'n_clusters': 9, 'n_init': 6}
Kmeans = KMeans(n_init = 6, algorithm='elkan', n_clusters = 2).fit(x_cs)
with open('KMeans.pkl','wb') as file:
    pickle.dump(Kmeans,file)
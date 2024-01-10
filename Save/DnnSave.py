import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle

data = pd.read_csv("Data/csgo_taskv2.csv")

data_major_true = data[data['bomb_planted'] == True]
data_major_false = data[data['bomb_planted'] == False]

data_major_false_undersampled = data_major_false.sample(len(data_major_true))
data1 = pd.concat([data_major_false_undersampled, data_major_true], axis = 0)

data1 = data1.drop(['map'], axis=1)
data1 = data1.drop(['Unnamed: 0'], axis=1)

y_cs = data1['bomb_planted']
x_cs = data1.drop(['bomb_planted'], axis=1)
x_train_cl, x_test_cl, y_train_cl, y_test_cl = train_test_split(x_cs, y_cs, test_size=0.3)


Dnn = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, activation="relu", input_shape=(14,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

Dnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy")
Dnn.fit(x_train_cl.astype(np.float64), y_train_cl.astype(np.float64), epochs=100, verbose=None)

Dnn.save('Dnn.h5')
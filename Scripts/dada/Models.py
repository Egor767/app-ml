import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import statistics

def ML1(x_test):
    st.header("Обучение с учителем")

    choices = ["DecisionTreeClassifier", "KNNClassifier"]
    choice = st.selectbox("Модель", choices)
    if not(choice is None):
        if choice == "DecisionTreeClassifier":
            st.subheader("DecisionTreeClassifier")
            with open('Models/DecisionTreeClassifier.pkl', 'rb') as file:
                model = pickle.load(file)
            Predict(model, x_test)
        
        elif choice == "KNNClassifier":
            st.subheader("KNNClassifier")
            with open('Models/KNN.pkl', 'rb') as file:
                model = pickle.load(file)
            Predict(model, x_test)


def ML2(x_test):
    st.header("Обучение без учителя")
    st.subheader("KMeans")

    with open('Models/KMeans.pkl', 'rb') as file:
        model = pickle.load(file)
    Predict(model, x_test)


def Ensembles(x_test):
    st.header("Ансамбли")

    choices = ["Stacking", "Bagging", "GradientBoosting"]
    choice = st.selectbox("Ансамбль", choices)
    if not(choice is None):
        if choice == "Stacking":
            st.subheader("Stacking")
            with open('Models/Stacking.pkl', 'rb') as file:
                model = pickle.load(file)
            Predict(model, x_test)
 
        elif choice == "Bagging":
            st.subheader("Bagging")
            with open('Models/Bagging.pkl', 'rb') as file:
                model = pickle.load(file)
            Predict(model, x_test)
        
        elif choice == "GradientBoosting":
            st.subheader("GradientBoosting")
            with open('Models/Boosting.pkl', 'rb') as file:
                model = pickle.load(file)
            Predict(model, x_test)


def Dnn(x_test):
    st.header("Нейронная сеть")
    st.subheader("Dnn")

    model = load_model('Models/Dnn.h5')
    PredictDnn(model, x_test)


def Predict(model, x_test):
    predict = model.predict(x_test)
    st.write("Класс объекта: ", predict[0])
    

def PredictDnn(model, x_test):
    predict = model.predict(x_test)
    st.write("Объект относится к 1 классу(True) с вероятностью: ", predict[0])


def DataLoader():
    data = st.file_uploader("Загрузите датасет в формате *.csv", type = "csv")
    if not(data is None):
        data = pd.read_csv(data)
    return data.drop(["Unnamed: 0", "map", "bomb_planted"], axis = 1)


def DataInput():
    data_input = []
    data_ex = pd.read_csv("Data/csgo_taskv2.csv")
    data_ex = data_ex.drop(["Unnamed: 0"], axis=1)
    
    names = data_ex.columns
    names = names.drop(["map", "bomb_planted"])
    for i in range(len(names)):
        inp = st.number_input(names[i], value = 1)
        data_input.append(inp)
    
    data_input = np.array(data_input)
    data_result = pd.DataFrame(data_input.reshape(1, -1), columns = names)
    
    st.write(data_result)
    return data_result


st.title("Модели Классификации")
st.header("Выбор типа предсказания классификации:")

choices = ["Для одного объекта", "Для датасета"]
choice = st.selectbox("Предсказание", choices)

if not(choice is None):
    if choice == "Для одного объекта":
        data = DataInput()
    elif choice == "Для датасета":
        data = DataLoader()

    check = st.checkbox('Выбрать модель')
    if check:
        choices2 = ["Обучение с учителем", "Обучение без учителя", "Ансамбли", "Нейронные сети"]
        choice2 = st.selectbox("Вид Модели", choices2)

        if not(choice2 is None):
            if choice2 == "Обучение с учителем":
                ML1(data)
            elif choice2 == "Обучение без учителя":
                ML2(data)
            elif choice2 == "Ансамбли":
                Ensembles(data)
            elif choice2 == "Нейронные сети":
                Dnn(data)

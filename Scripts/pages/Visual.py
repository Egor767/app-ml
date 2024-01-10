import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


def HeatMap():
    copy = data.drop(["map"], axis=1)
    st.header("Тепловая карта")
    plt.figure(figsize=(16,10))
    sns.heatmap(copy.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)


def Hist():
    st.header("Гистограмма распределения")
    features = data.columns.to_list()
    select = st.selectbox("Признак", features, key = 'selectbox1')
    plt.figure(figsize=(12,7))
    sns.histplot(data[select])
    st.pyplot(plt)


def BoxPlot():
    st.header("Диаграмма Ящик с усами")
    features = data.columns.to_list()
    select = st.selectbox("Признак", features)
    plt.figure(figsize=(12,7))
    sns.boxplot(x = data[select])
    st.pyplot(plt)


def Scatter():
    st.header("Диаграмма рассеяния")
    features = data.columns.to_list()
    select1 = st.selectbox("Первый признак", features, key='selectbox1')
    select2 = st.selectbox("Второй признак", features, key='selectbox2')
    plt.figure(figsize=(12,7))
    plt.scatter(data[select1][:500], data[select2][:500], )
    st.pyplot(plt)


data = pd.read_csv("Data/csgo_taskv2.csv")
data = data.drop(["Unnamed: 0"], axis=1)

st.title("Выберите визуализацию:")
visual_types = ['Гистограмма распределения', 'Ящик с усами', 'Диаграмма рассеяния', 'Тепловая карта']

select = st.selectbox("Визуализация", visual_types)

if not(select is None):
    if select == "Тепловая карта":
        HeatMap()
    elif select == "Диаграмма рассеяния":
        Scatter()
    elif select == "Гистограмма распределения":
        Hist()
    elif select == "Ящик с усами":
        BoxPlot()

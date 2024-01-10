import streamlit as st
from PIL import Image

st.title("Разработка Web-приложения для инференса моделей ML и анализа данных")

st.header("Автор работы:")


st.subheader("Мусияк Егор Алексеевич")
st.divider()
st.subheader("Студент группы МО-221")
st.divider()
st.subheader("Фото")
img = Image.open("image.jpg")
st.image(img, width=250)


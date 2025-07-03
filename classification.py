import streamlit as st
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression

@st.cache_data
def load_data():
    iris=load_iris()
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    df['Species']=iris.target
    target_names = iris.target_names
    return df , target_names


df,target_names = load_data()

model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['Species'])



st.sidebar.title("Input Features")
sepal_length = st.sidebar.slider("sepal Length",float(df['sepal length (cm)'].min()),float(df["sepal length (cm)"].max()))

st.sidebar.title("Input Features")
sepal_width = st.sidebar.slider("sepal width",float(df['sepal width (cm)'].min()),float(df["sepal width (cm)"].max()))

st.sidebar.title("Input Features")
petal_length = st.sidebar.slider("petal Length",float(df['petal length (cm)'].min()),float(df["petal length (cm)"].max()))

st.sidebar.title("Input Features")
petal_width = st.sidebar.slider("petal width",float(df['petal width (cm)'].min()),float(df["petal width (cm)"].max()))


input_data = [[sepal_length,sepal_width, petal_length , petal_width]]


# prediction

prediction = model.predict(input_data)
predicted_species= target_names[prediction[0]]

st.write("Prediction")
st.write(f"The predicted species is {predicted_species}")
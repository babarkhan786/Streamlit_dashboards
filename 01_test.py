import streamlit as st
import pandas as pd
import seaborn as sns

st.header("This video is brought to you by AD-Babar")
st.text("Aap aaj kia plan kar rahy")

df = sns.load_dataset('iris')
st.write(df[['species', 'sepal_length', 'petal_length']].head(10))

st.bar_chart(df['sepal_length'])
st.line_chart(df['sepal_length'])
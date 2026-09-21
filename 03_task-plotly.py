# import libraries
import streamlit as st 
import plotly.express as px 
import pandas as pd

# import dataset
st.title("Plotly and Streamlit ko bana k app bnani hy")

df = px.data.iris()

st.write(df.head())
st.write(df.columns)

# summary statistics
st.write(df.describe())

# data management
species_option = df["species"].unique().tolist()

species = st.selectbox(
    "Which species should we plot?",
    species_option,
    0
)

# filter data
df = df[df["species"] == species]

st.write(df)

# import libraries
import streamlit as st
import plotly.express as px
import pandas as pd

# import dataset
st.title("Iris Flower Classification")

df = px.data.iris()

# Display data
st.subheader("Iris Dataset")
st.write(df.head())
st.write("Species:", df["species"].unique().tolist())

# Classification plot: all species
st.subheader("All Iris Species")

fig1 = px.scatter(
    df,
    x="sepal_width",
    y="sepal_length",
    color="species",
    symbol="species",
    hover_name="species",
    title="Sepal Width vs Sepal Length",
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig1.update_layout(width=800, height=500)

st.plotly_chart(fig1, use_container_width=True)


# Scatter plot: all species
st.subheader("Petal Measurements - All Species")

fig2 = px.scatter(
    df,
    x="petal_width",
    y="petal_length",
    color="species",
    size="sepal_length",
    hover_name="species",
    title="Petal Width vs Petal Length",
    color_discrete_sequence=px.colors.qualitative.Vivid
)

fig2.update_layout(width=800, height=500)

st.plotly_chart(fig2, use_container_width=True)

# 3D scatter plot
st.subheader("Iris Species - 3D Scatter Plot")

fig = px.scatter_3d(
    df,
    x="sepal_length",
    y="sepal_width",
    z="petal_length",
    color="species",
    hover_name="species",
    title="3D Iris Species Visualization",
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig.update_layout(width=900, height=600)

st.plotly_chart(fig, use_container_width=True)

# Violin plot
st.subheader("Iris Species - Violin Plot")

fig = px.violin(
    df,
    x="species",
    y="petal_length",
    color="species",
    box=True,
    title="Petal Length Distribution by Species",
    color_discrete_sequence=px.colors.qualitative.Vivid
)

fig.update_layout(width=800, height=500)

st.plotly_chart(fig, use_container_width=True)

# Box plot
st.subheader("Iris Species - Box Plot")

fig = px.box(
    df,
    x="species",
    y="petal_length",
    color="species",
    title="Petal Length Distribution by Species",
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig.update_layout(width=800, height=500)

st.plotly_chart(fig, use_container_width=True)
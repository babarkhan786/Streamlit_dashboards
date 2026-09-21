# import libraries
import streamlit as st 
import plotly.express as px 
import pandas as pd

# import dataset
st.title ('Plotly and streaamlit ko bna k app bnani hy')
df = px.data.gapminder()
st.write()

st.write(df.head())

st.write(df.columns)

# write summary stat
st.write(df.describe())

# data mangmnet
year_option = df['year'].unique().tolist()
year = st.selectbox("Which year should we plot", year_option, 0)
# df = df[df['year']==year]

# plotting
fig = px.scatter(df, x= 'gdpPercap', y='lifeExp', color='country', hover_name='country', size='pop',
                 log_x=True, size_max=55, range_x=[100, 100000], range_y=[20, 90],
                 animation_frame='year', animation_group='country')

fig.update_layout(width=800, height=400)
st.write(fig)
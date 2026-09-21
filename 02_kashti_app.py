from pyexpat import features
import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# make containers
header = st.container()
data_sets = st.container()
features = st. container()
model_training = st.container()

with header:
    st.title("kashti ki app")
    st.text("In the project we will work on kashti data")
    
with data_sets:
    st.header("kashti doob gaye")
    st.text("we will work on titanic dataset")
    # import data
    df = sns.load_dataset('titanic')
    df = df.dropna()
    
    st.write(df.head(10))
    
    st.subheader("kitny log thy")
    st.bar_chart(df['sex'].value_counts())
    
    # other plots
    st.subheader("class k hisab say farq")
    st.bar_chart(df['class'].value_counts())
    
    #bar plot
    st.bar_chart(df['age'].sample(10))
    
with features:
    st.header("These are our app features")
    st.text("Exploring kashti data")
    st.markdown('1. **features 1:**')
    st.markdown('2. **features 2:**')

with model_training:
    st.header("Kashti walon k sath kia hova")
    st.text("training the model")
    
    # making columns
    input, display = st.columns(2)
    
    # slider
    max_septh = input.slider("how many people do you know?", 
                             min_value = 5, max_value=100, value = 5, step=5)
# n estimaters
n_estimaters = input.selectbox("How many trees should be therein RF?", 
                               options=[50,100,200,300, 'NO limit'])


# input features from user

input_feature = input.selectbox(
    "Which feature should we use?",
    options=["age", "fare", "pclass", "sibsp", "parch"]
)

target = 'fare'

# machine learning model
model = RandomForestRegressor(
    max_depth=max_septh,
    n_estimators=n_estimaters,
    random_state=42)
# define X and y

X = df[[input_feature]]
y = df[[target]]

# fit the model
model.fit(X,y)

#prediction
pred = model.predict(X)

#metrics
mae = mean_absolute_error(y, pred)
mse = mean_squared_error(y, pred)
r2 = r2_score(y, pred)

# Display results
display.subheader("Model Results")

display.write("Selected feature:")
display.write(input_feature)

display.write("MAE:")
display.write(mae)

display.write("MSE:")
display.write(mse)

display.write("R²:")
display.write(r2)
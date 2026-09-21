import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
# webapp title
st.markdown('''
    # **Exploratory Data Analysis web Application**
    This app is developed by Dr. AD-Babar celled **EDA app**''')
# how to upload file from pc
with st.sidebar.header("Upload your dataset (.csv)"):
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=['csv'])
    df = sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)")
# profiling report for ydata
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Profiling Report ydata**')
    st_profile_report(pr)
else:
    st.info('Awaiting for CSV file')
    if st.button('Press to use example data'):
        # eample data
        @st.cache
        def load_data():
            a = pd.DataFrame(np.random.rand(100,5),
                             columns=['age','bat','cat','Dutchland','Ear'])
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Profiling Report ydata**')
        st_profile_report(pr)
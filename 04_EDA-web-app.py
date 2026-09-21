import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
#Webapp ka title
st.markdown(''' 
            # **Exploratory Data Analysis web Application**
            This app is developed by Ad-Babar **EDA app**
            ''')
# how to upload from pc
with st.sidebar.header("Upload your dataset (.csv)"):
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=['csv'])
    df = sns.load_dataset('titanic')
    # Example CSV link
st.sidebar.markdown("[Example CSV file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)"
)


# profiling report for pandas
if uploaded_file is not None:
    @st.cache 
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, title=f"EDA Profiling Report - {df}", explorative=True)
    st.header('**Input DF**')
    st.write(df)
    st.write('---')
    st.header('**Profiling report with pandas**')
    st_profile_report(pr)
else:
    st.info("No CSV uploaded. Select an example dataset below.")
    if st.button('Press to use example data'):
    # example dataset
        @st.cache
        def load_data():
            a = pd.DataFrame(np.random.rand(100, 5),
                         columns=['age', 'banana','coding','Dutchland', 'Egg'])
            return a
        # Load selected dataset
        df = load_data()
        pr = ProfileReport(df, title=f"EDA Profiling Report - {df}", explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profilling Report**')
        st_profile_report(pr)

#Direct import from Kaggle when no local file is uploaded.
import os
import zipfile
import tempfile
from kaggle.api.kaggle_api_extended import KaggleApi

# --------------------------------------------------
# Kaggle Search
# --------------------------------------------------

st.sidebar.subheader("Kaggle Dataset")

search_term = st.sidebar.text_input(
    "Search Kaggle datasets",
    value=""
)


@st.cache_data
def search_kaggle_datasets(search_term):

    api = KaggleApi()
    api.authenticate()

    datasets = api.dataset_list(
        search=search_term,
        file_type="csv"
    )

    results = []

    for dataset in datasets:

        results.append({
            "title": dataset.title,
            "ref": dataset.ref
        })

    return results


# --------------------------------------------------
# Search button
# --------------------------------------------------

if st.sidebar.button("Search Kaggle"):

    if search_term.strip() == "":
        st.sidebar.warning(
            "Please enter a search term."
        )

    else:

        try:

            results = search_kaggle_datasets(
                search_term
            )

            if results:

                st.session_state["kaggle_results"] = results

                # Reset previously selected dataset
                st.session_state.pop(
                    "selected_dataset",
                    None
                )

                st.sidebar.success(
                    f"{len(results)} datasets found!"
                )

            else:

                st.session_state.pop(
                    "kaggle_results",
                    None
                )

                st.sidebar.warning(
                    "No datasets found."
                )

        except Exception as e:

            st.sidebar.error(
                f"Kaggle error: {e}"
            )
# --------------------------------------------------
# Select Kaggle dataset
# --------------------------------------------------

dataset_ref = None

if "kaggle_results" in st.session_state:

    results = st.session_state["kaggle_results"]

    dataset_options = [
        f"{item['title']} | {item['ref']}"
        for item in results
    ]

    selected_dataset = st.sidebar.selectbox(
        "Select Kaggle dataset",
        dataset_options,
        key="selected_dataset"
    )

    selected_index = dataset_options.index(
        selected_dataset
    )

    dataset_ref = results[selected_index]["ref"]


# --------------------------------------------------
# Download Kaggle dataset
# --------------------------------------------------

@st.cache_data
def download_kaggle_dataset(dataset_ref):

    api = KaggleApi()
    api.authenticate()

    # Create temporary directory
    download_dir = os.path.join(
        tempfile.gettempdir(),
        "streamlit_kaggle_data",
        dataset_ref.replace("/", "_")
    )

    os.makedirs(
        download_dir,
        exist_ok=True
    )

    # Download dataset
    api.dataset_download_files(
        dataset_ref,
        path=download_dir,
        unzip=True
    )

    # Find CSV files
    csv_files = []

    for root, dirs, files in os.walk(download_dir):

        for file in files:

            if file.lower().endswith(".csv"):

                csv_files.append(
                    os.path.join(root, file)
                )

    if not csv_files:

        raise ValueError(
            "No CSV file found in this Kaggle dataset."
        )

    return csv_files


# --------------------------------------------------
# Load selected CSV from Kaggle
# --------------------------------------------------

@st.cache_data
def load_csv_file(csv_file):

    return pd.read_csv(csv_file)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = None
source = None
dataset_name = None

# Local file has priority
if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        source = "Local CSV"
        dataset_name = uploaded_file.name

        st.success(
            f"Dataset loaded: {dataset_name}"
        )

    except Exception as e:

        st.error(
            f"Could not read CSV file: {e}"
        )
# Kaggle dataset
elif dataset_ref is not None:

    try:

        csv_files = download_kaggle_dataset(
            dataset_ref
        )

        # Create simple names for the selectbox
        csv_names = [
            os.path.relpath(
                file,
                os.path.join(
                    tempfile.gettempdir(),
                    "streamlit_kaggle_data",
                    dataset_ref.replace("/", "_")
                )
            )
            for file in csv_files
        ]

        # Select CSV if dataset contains multiple files
        selected_csv_name = st.sidebar.selectbox(
            "Select CSV file",
            csv_names
        )

        selected_csv_index = csv_names.index(
            selected_csv_name
        )

        selected_csv = csv_files[
            selected_csv_index
        ]

        df = load_csv_file(
            selected_csv
        )

        source = f"Kaggle: {dataset_ref}"
        dataset_name = selected_csv_name
        st.success(
            f"Dataset loaded: {dataset_ref}"
        )

    except Exception as e:

        st.error(
            f"Could not load Kaggle dataset: {e}"
        )


# --------------------------------------------------
# No Dataset
# --------------------------------------------------

else:

    st.info(
        "Upload a CSV file or search for a Kaggle dataset."
    )


# --------------------------------------------------
# Display Dataset
# --------------------------------------------------

if df is not None:

    st.header("Input Dataset")

    # Dataset information
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )
    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )
    st.header("Dataset Overview")

    st.subheader(
        f"📊 Dataset: {dataset_name}"
    )
    
    st.caption(
        f"Data source: {source}"
    )

    # Preview
    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.write("---")

    # --------------------------------------------------
    # Download CSV
    # --------------------------------------------------

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Dataset",
        data=csv_data,
        file_name="selected_dataset.csv",
        mime="text/csv"
    )

    st.write("---")


    # --------------------------------------------------
    # Profiling Report
    # --------------------------------------------------

    st.header(
        "Profiling Report"
    )

    if st.button(
        "Generate Profiling Report"
    ):

            pr = ProfileReport(
                df,
                title=f"EDA Profiling Report - {dataset_name}",
                explorative=True
             
            )

            st_profile_report(pr)
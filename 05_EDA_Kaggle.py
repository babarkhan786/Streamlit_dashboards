import os
import tempfile

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from kaggle.api.kaggle_api_extended import KaggleApi
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="EDA Web Application",
    page_icon="📊",
    layout="wide"
)


# ==================================================
# TITLE
# ==================================================

st.markdown("""
# **Exploratory Data Analysis Web Application**

This app is developed by **Ad-Babar**
""")


# ==================================================
# SIDEBAR: LOCAL CSV
# ==================================================

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


# ==================================================
# SIDEBAR: KAGGLE SEARCH
# ==================================================

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


# ==================================================
# KAGGLE SEARCH BUTTON
# ==================================================

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


# ==================================================
# SELECT KAGGLE DATASET
# ==================================================

dataset_ref = None

if "kaggle_results" in st.session_state:

    results = st.session_state["kaggle_results"]

    dataset_options = [
        f"{item['title']} | {item['ref']}"
        for item in results
    ]

    selected_dataset = st.sidebar.selectbox(
        "Select Kaggle dataset",
        dataset_options
    )

    selected_index = dataset_options.index(
        selected_dataset
    )

    dataset_ref = results[selected_index]["ref"]


# ==================================================
# DOWNLOAD KAGGLE DATASET
# ==================================================

@st.cache_data
def download_kaggle_dataset(dataset_ref):

    api = KaggleApi()
    api.authenticate()

    download_dir = os.path.join(
        tempfile.gettempdir(),
        "streamlit_kaggle_data",
        dataset_ref.replace("/", "_")
    )

    os.makedirs(
        download_dir,
        exist_ok=True
    )

    api.dataset_download_files(
        dataset_ref,
        path=download_dir,
        unzip=True
    )

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


# ==================================================
# LOAD CSV
# ==================================================

@st.cache_data
def load_csv_file(csv_file):

    return pd.read_csv(csv_file)


# ==================================================
# LOAD DATASET
# ==================================================

df = None
source = None


# --------------------------------------------------
# Local CSV
# --------------------------------------------------

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        source = "Local CSV"

        st.success(
            "Uploaded CSV dataset loaded!"
        )

    except Exception as e:

        st.error(
            f"Could not read CSV file: {e}"
        )


# --------------------------------------------------
# Kaggle CSV
# --------------------------------------------------

elif dataset_ref is not None:

    try:

        csv_files = download_kaggle_dataset(
            dataset_ref
        )

        base_dir = os.path.join(
            tempfile.gettempdir(),
            "streamlit_kaggle_data",
            dataset_ref.replace("/", "_")
        )

        csv_names = [
            os.path.relpath(
                file,
                base_dir
            )
            for file in csv_files
        ]

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

        st.success(
            f"Kaggle dataset loaded: {dataset_ref}"
        )

    except Exception as e:

        st.error(
            f"Could not load Kaggle dataset: {e}"
        )


# ==================================================
# NO DATASET
# ==================================================

else:

    st.info(
        "Upload a CSV file or search for a Kaggle dataset."
    )


# ==================================================
# EDA
# ==================================================

if df is not None:

    # ==================================================
    # DATASET OVERVIEW
    # ==================================================

    st.header("Dataset Overview")

    st.caption(
        f"Data source: {source}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Missing Values",
            f"{df.isna().sum().sum():,}"
        )

    with col4:

        st.metric(
            "Duplicate Rows",
            f"{df.duplicated().sum():,}"
        )


    # ==================================================
    # DATA PREVIEW
    # ==================================================

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )


    # ==================================================
    # DATA TYPES
    # ==================================================

    with st.expander("Data Types"):

        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing": df.isna().sum().values,
            "Unique Values": [
                df[col].nunique()
                for col in df.columns
            ]
        })

        st.dataframe(
            dtype_df,
            use_container_width=True
        )


    # ==================================================
    # MISSING VALUES
    # ==================================================

    with st.expander("Missing Values Analysis"):

        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isna().sum().values,
            "Missing %": (
                df.isna().mean().values * 100
            )
        })

        missing_df = missing_df[
            missing_df["Missing Values"] > 0
        ]

        if missing_df.empty:

            st.success(
                "No missing values found."
            )

        else:

            st.dataframe(
                missing_df.sort_values(
                    "Missing Values",
                    ascending=False
                ),
                use_container_width=True
            )


    # ==================================================
    # DESCRIPTIVE STATISTICS
    # ==================================================

    st.header("Descriptive Statistics")

    numeric_df = df.select_dtypes(
        include="number"
    )

    if not numeric_df.empty:

        st.dataframe(
            numeric_df.describe().T,
            use_container_width=True
        )

    else:

        st.info(
            "No numeric columns available."
        )


    # ==================================================
    # STATISTICS BY DATA TYPE
    # ==================================================

    st.subheader("Dataset Summary")

    summary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Non-Null": df.notna().sum(),
        "Missing": df.isna().sum(),
        "Unique": df.nunique()
    })

    st.dataframe(
        summary,
        use_container_width=True
    )


    # ==================================================
    # VISUALIZATIONS
    # ==================================================

    st.header("Data Visualizations")


    # --------------------------------------------------
    # Numeric Columns
    # --------------------------------------------------

    numeric_columns = list(
        numeric_df.columns
    )

    if numeric_columns:

        st.subheader("Numeric Variable Analysis")

        selected_numeric = st.selectbox(
            "Select numeric variable",
            numeric_columns
        )


        # ----------------------------------------------
        # Histogram
        # ----------------------------------------------

        st.markdown(
            f"### Distribution of `{selected_numeric}`"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.histplot(
            df[selected_numeric].dropna(),
            kde=True,
            ax=ax
        )

        ax.set_xlabel(
            selected_numeric
        )

        ax.set_ylabel(
            "Frequency"
        )

        st.pyplot(fig)

        plt.close(fig)


        # ----------------------------------------------
        # Box Plot
        # ----------------------------------------------

        st.markdown(
            f"### Box Plot of `{selected_numeric}`"
        )

        fig, ax = plt.subplots(
            figsize=(10, 3)
        )

        sns.boxplot(
            x=df[selected_numeric],
            ax=ax
        )

        ax.set_xlabel(
            selected_numeric
        )

        st.pyplot(fig)

        plt.close(fig)


        # ----------------------------------------------
        # Numeric Statistics
        # ----------------------------------------------

        st.markdown(
            f"### Statistics for `{selected_numeric}`"
        )

        stats = df[selected_numeric].describe()

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        with stat_col1:

            st.metric(
                "Mean",
                f"{df[selected_numeric].mean():.2f}"
            )

        with stat_col2:

            st.metric(
                "Median",
                f"{df[selected_numeric].median():.2f}"
            )

        with stat_col3:

            st.metric(
                "Minimum",
                f"{df[selected_numeric].min():.2f}"
            )

        with stat_col4:

            st.metric(
                "Maximum",
                f"{df[selected_numeric].max():.2f}"
            )


    # ==================================================
    # CORRELATION ANALYSIS
    # ==================================================

    if len(numeric_columns) >= 2:

        st.subheader("Correlation Heatmap")

        correlation = numeric_df.corr()

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax
        )

        ax.set_title(
            "Correlation Matrix"
        )

        st.pyplot(fig)

        plt.close(fig)


        # ----------------------------------------------
        # Scatter Plot
        # ----------------------------------------------

        st.subheader("Scatter Plot")

        col1, col2 = st.columns(2)

        with col1:

            x_variable = st.selectbox(
                "X variable",
                numeric_columns,
                key="x_variable"
            )

        with col2:

            y_variable = st.selectbox(
                "Y variable",
                numeric_columns,
                index=min(
                    1,
                    len(numeric_columns) - 1
                ),
                key="y_variable"
            )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.scatterplot(
            data=df,
            x=x_variable,
            y=y_variable,
            ax=ax
        )

        ax.set_title(
            f"{x_variable} vs {y_variable}"
        )

        st.pyplot(fig)

        plt.close(fig)


    # ==================================================
    # CATEGORICAL ANALYSIS
    # ==================================================

    categorical_columns = list(
        df.select_dtypes(
            exclude="number"
        ).columns
    )

    if categorical_columns:

        st.subheader(
            "Categorical Variable Analysis"
        )

        selected_category = st.selectbox(
            "Select categorical variable",
            categorical_columns
        )

        value_counts = (
            df[selected_category]
            .value_counts()
            .head(20)
        )

        st.dataframe(
            value_counts.rename(
                "Count"
            ).to_frame(),
            use_container_width=True
        )


        # ----------------------------------------------
        # Count Plot
        # ----------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.countplot(
            data=df,
            y=selected_category,
            order=value_counts.index,
            ax=ax
        )

        ax.set_title(
            f"Distribution of {selected_category}"
        )

        ax.set_xlabel(
            "Count"
        )

        ax.set_ylabel(
            selected_category
        )

        st.pyplot(fig)

        plt.close(fig)


    # ==================================================
    # GROUPED STATISTICS
    # ==================================================

    if categorical_columns and numeric_columns:

        st.subheader(
            "Grouped Statistics"
        )

        group_column = st.selectbox(
            "Group by",
            categorical_columns,
            key="group_column"
        )

        value_column = st.selectbox(
            "Numeric variable",
            numeric_columns,
            key="group_value"
        )

        grouped = (
            df.groupby(group_column)[value_column]
            .agg(
                Count="count",
                Mean="mean",
                Median="median",
                Minimum="min",
                Maximum="max",
                Std="std"
            )
            .sort_values(
                "Mean",
                ascending=False
            )
        )

        st.dataframe(
            grouped,
            use_container_width=True
        )


    # ==================================================
    # DOWNLOAD DATASET
    # ==================================================

    st.header("Download Dataset")

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="selected_dataset.csv",
        mime="text/csv"
    )


    # ==================================================
    # YDATA PROFILING
    # ==================================================

    st.header("Automated Profiling Report")

    st.info(
        "The manual EDA sections above provide statistics and plots. "
        "The automated profiling report is optional."
    )

    if st.button(
        "Generate Profiling Report"
    ):

        with st.spinner(
            "Generating profiling report..."
        ):

            try:

                pr = ProfileReport(
                    df,
                    title="EDA Profiling Report",
                    explorative=True
                )

                st_profile_report(pr)

            except Exception as e:

                st.error(
                    f"Profiling report could not be generated: {e}"
                )
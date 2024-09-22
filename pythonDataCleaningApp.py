import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import io  # Required for in-memory file handling

# Title of the dashboard
st.title("Data Cleaning Dashboard")

# File Upload
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xls", "xlsx", "json"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1]

    if file_type == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_type in ['xls', 'xlsx']:
        df = pd.read_excel(uploaded_file)
    elif file_type == 'json':
        df = pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file type!")

    st.write("### Data Preview")
    st.dataframe(df.head())

    # 1. Data Summary
    if st.checkbox("Show Data Summary"):
        st.write("### General Summary:")
        st.write(f"Number of entries: {len(df)}")
        st.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        st.write("Data Types:", df.dtypes)
        st.write(df.describe())

    # 2. Handling Missing Values
    if st.checkbox("Handle Missing Values"):
        st.write("### Missing Values Summary:")
        st.write(df.isnull().sum())
        method = st.selectbox("Fill missing values with:", ("None", "Mean", "Median", "Mode"))

        if method == "Mean":
            df.fillna(df.mean(), inplace=True)
        elif method == "Median":
            df.fillna(df.median(), inplace=True)
        elif method == "Mode":
            df.fillna(df.mode().iloc[0], inplace=True)

        st.write("### Data After Handling Missing Values:")
        st.dataframe(df)

    # 3. Handling Duplicates
    if st.checkbox("Remove Duplicates"):
        df.drop_duplicates(inplace=True)
        st.write("### Data After Removing Duplicates:")
        st.dataframe(df)

    # 4. Outlier Detection
    if st.checkbox("Detect and Remove Outliers"):
        numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
        col = st.selectbox("Select column for outlier detection", numerical_columns)
        if col:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
            st.write(f"Outliers removed based on IQR for column {col}")
            st.dataframe(df)

    # 5. Normalization and Scaling
    if st.checkbox("Normalize/Scale Data"):
        method = st.selectbox("Choose normalization method:", ["None", "Z-score", "MinMax", "Robust"])
        columns = st.multiselect("Select columns to normalize/scale", df.select_dtypes(include=np.number).columns)

        if method == "Z-score":
            df[columns] = StandardScaler().fit_transform(df[columns])
        elif method == "MinMax":
            df[columns] = MinMaxScaler().fit_transform(df[columns])
        elif method == "Robust":
            df[columns] = RobustScaler().fit_transform(df[columns])

        st.write("### Data After Normalization/Scaling")
        st.dataframe(df)

    # 6. Text Data Consistency
    if st.checkbox("Clean Text Data"):
        text_columns = df.select_dtypes(include='object').columns.tolist()
        column = st.selectbox("Select text column to clean", text_columns)
        action = st.selectbox("Choose action:", ["Lowercase", "Uppercase", "Capitalize"])

        if action == "Lowercase":
            df[column] = df[column].str.lower().str.strip()
        elif action == "Uppercase":
            df[column] = df[column].str.upper().str.strip()
        elif action == "Capitalize":
            df[column] = df[column].str.capitalize().str.strip()

        st.write(f"### Data After Text Cleaning ({action})")
        st.dataframe(df)

    # 7. Exporting Cleaned Data
    st.write("### Download Cleaned Data")
    file_format = st.selectbox("Choose file format for download", ["CSV", "Excel", "JSON"])

    if file_format == "CSV":
        # Convert dataframe to CSV and encode to bytes for download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name='cleaned_data.csv', mime='text/csv')

    elif file_format == "Excel":
        # Save Excel file to an in-memory buffer
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        excel_data = output.getvalue()  # Get the data in the buffer as bytes
        st.download_button(label="Download Excel", data=excel_data, file_name='cleaned_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    elif file_format == "JSON":
        # Convert dataframe to JSON and encode to bytes for download
        json_data = df.to_json().encode('utf-8')
        st.download_button(label="Download JSON", data=json_data, file_name='cleaned_data.json', mime='application/json')

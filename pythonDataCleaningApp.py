import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from thefuzz import fuzz, process #For fuzzy matching
import io  # Required for in-memory file handling

#Set Wide Layout
st.set_page_config(layout="wide")


# Title of the dashboard
st.title("Data Cleaning Dashboard")

# Initialize session state to store converted column descriptions
if 'converted_columns' not in st.session_state:
    st.session_state.converted_columns = {}

# File Upload
uploaded_file = st.file_uploader("Upload File for Data Cleaning", type=["csv", "xls", "xlsx", "json"])

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
        
    st.write("### Convert Text Data to Lowercase")
    # 0. Option to convert text data to lowercase at the beginning
    if st.checkbox("Lowercasing Text Data"):
        text_columns = df.select_dtypes(include='object').columns.tolist()
        if text_columns:
            column = st.selectbox("Select text column to convert to lowercase", text_columns)
            if st.button("Convert to Lowercase"):
                df[column] = df[column].str.lower().str.strip()
                st.success(f"Converted column '{column}' to lowercase.")
                st.dataframe(df)  # Display the modified DataFrame


    # 1. Data Summary for Numerical Columns
    st.write("### Showing Data Summary")
    if st.checkbox("Data Summary"):
        
        with st.expander("### Data Preview"):
            st.dataframe(df.head())

        st.write("### General Summary")
        st.write(f"Number of entries: {len(df)}")
        st.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        with st.expander("### View Data Types"):
            df_types = pd.DataFrame(df.dtypes).reset_index()
            df_types.columns = ['Column Name', 'Data Type']
            st.dataframe(df_types)

        # Display summary for numerical columns only
        with st.expander("### Statistics for Columns with Numerical Data Types:"):
            summary_df = pd.DataFrame(df.describe()).reset_index()
            summary_df.columns = ['Statistics', 'Values']
            st.dataframe(summary_df)
        
        
        # Handling unsupported data types
        unsupported_columns = [col for col in df.columns if df[col].dtype in ['object', 'complex', 'mixed', 'timedelta64[ns]', 'sparse']]

        # Handle unsupported columns
        if unsupported_columns:
            st.warning(f"The Following Columns Contain Unsupported Data Types: {unsupported_columns}")
            col_to_convert = st.selectbox("Select a Column to Convert to a Supported Data Type:", unsupported_columns)

            conversion_method = st.selectbox("Choose a Supported Data Type:", ["To Numeric", "To String", "Drop Column", "None"])

            if st.button("Convert & Describe"):
                if conversion_method == "To Numeric":
                    # Convert to numeric; non-convertible values will be replaced with NaN
                    df[col_to_convert] = pd.to_numeric(df[col_to_convert], errors='coerce')
                    st.success(f"Converted column '{col_to_convert}' to numeric.")
                elif conversion_method == "To String":
                    # Convert to string
                    df[col_to_convert] = df[col_to_convert].astype(str)
                    st.success(f"Converted column '{col_to_convert}' to string.")
                elif conversion_method == "Drop Column":
                    # Drop the column
                    df.drop(columns=[col_to_convert], inplace=True)
                    st.success(f"Dropped column '{col_to_convert}' from the DataFrame.")

                # Store the description of the converted column in session state
                st.session_state.converted_columns[col_to_convert] = df[col_to_convert].describe(include='all')

        # Display tables for all converted columns stored in session state
        if st.session_state.converted_columns:
            with st.expander("### Summary Statistics for Converted Columns:"):
                cols = st.columns(len(st.session_state.converted_columns))
                for i, (col_name, summary) in enumerate(st.session_state.converted_columns.items()):
                    with cols[i]:
                        st.write(f"**{col_name}**")
                        
                        summary_df = pd.DataFrame(summary).reset_index()
                        summary_df.columns = ['Statistics', 'Values']  # Rename columns
                        st.dataframe(summary_df)

    # 2. Handling Missing Values
    st.write("### Handling Missing Values")
    if st.checkbox("Missing Values"):
        with st.expander("### Missing Values Summary:"):
            missing_values_df = df.isnull().sum().add((df == '').sum()).add(df.isin([-9999, -1, 0, 'missing', 'na', 'not applicable']).sum()).reset_index()
            missing_values_df.columns = ['Column Name', 'Number of Missing Entries']
            st.dataframe(missing_values_df)

        method = st.selectbox("Fill missing values with:", ("None", "Mean", "Median", "Mode"))

        if method == "Mean":
            df.fillna(df.mean(), inplace=True)
        elif method == "Median":
            df.fillna(df.median(), inplace=True)
        elif method == "Mode":
            df.fillna(df.mode().iloc[0], inplace=True)

        with st.expander("### Data After Handling Missing Values:"):
            st.dataframe(df)

    # 3. Handling Duplicates
    st.write("### Handling Duplicates")
    if st.checkbox("Duplicates"):
        # a) Identifying Duplicates
        st.write("##### Identifying Duplicates")
        
        # i) Exact Matches
        if st.checkbox("Identifying Exact Duplicates"):
            exact_dup = df[df.duplicated(keep=False)]
            st.write(f"**Exact Duplicates**: {exact_dup.shape[0]} entries found")
            with st.expander("Showing Exact Duplicates"):
                st.dataframe(exact_dup)

        # ii) Fuzzy Matching
        if st.checkbox("Identifying Almost Similar Duplicates"):
            # Select column for fuzzy matching
            column_to_match = st.selectbox("Select column for fuzzy matching", df.columns)
            threshold = st.slider("Set similarity threshold for fuzzy matching", 70, 100, 90)
            matches = []
            for i, val in enumerate(df[column_to_match]):
                for j in range(i + 1, len(df)):
                    score = fuzz.ratio(val, df[column_to_match].iloc[j])
                    if score >= threshold:
                        matches.append((i, j, val, df[column_to_match].iloc[j], score))

            fuzzy_dup_df = pd.DataFrame(matches, columns=["Row 1", "Row 2", "Value 1", "Value 2", "Score"])
            st.write(f"**Fuzzy Matches**: {fuzzy_dup_df.shape[0]} pairs found")
            with st.expander("Showing Fuzzy Matches"):
                st.dataframe(fuzzy_dup_df)
                
        # b) Handling Duplicates
        st.write("##### Removing Duplicates")

        # i) Removing Duplicates
        if st.checkbox("Removing Exact Duplicates"):
            df.drop_duplicates(inplace=True)
            st.success("Exact duplicates removed.")
            with st.expander("Data After Removing Duplicates"):
                st.dataframe(df)

        #ii) Aggregating Data
        if st.checkbox("Aggregating Duplicates"):
            agg_column = st.selectbox("Select column to aggregate duplicate data", df.columns)
            agg_method = st.selectbox("Choose aggregation method", ["sum", "mean", "count", "first", "last"])
            
            if agg_method == "sum":
                df_agg = df.groupby(agg_column).sum()
            elif agg_method == "mean":
                df_agg = df.groupby(agg_column).mean()
            elif agg_method == "count":
                df_agg = df.groupby(agg_column).count()
            elif agg_method == "first":
                df_agg = df.groupby(agg_column).first()
            elif agg_method == "last":
                df_agg = df.groupby(agg_column).last()
            
            st.success(f"Data aggregated using '{agg_method}' method.")
            with st.expander("Aggregated Data"):
                st.dataframe(df_agg)


    # 4. Outlier Detection
    st.write("### Handling Outliers")
    if st.checkbox("Outlier Detection"):
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
    st.write("### Normalizing Data")
    if st.checkbox("Normalization"):
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
    st.write("### Restoring Text Data Consistency")
    if st.checkbox("Text Data Consistency"):
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
    st.write("### Downloading Cleaned Data")
    file_format = st.selectbox("Choose file format for download", ["CSV", "Excel", "JSON"])

    if file_format == "CSV":
        # Convert dataframe to CSV and encode to bytes for download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV File", data=csv, file_name='cleaned_data.csv', mime='text/csv')

    elif file_format == "Excel":
        # Save Excel file to an in-memory buffer
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        excel_data = output.getvalue()  # Get the data in the buffer as bytes
        st.download_button(label="Download Excel File", data=excel_data, file_name='cleaned_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    elif file_format == "JSON":
        # Convert dataframe to JSON and encode to bytes for download
        json_data = df.to_json().encode('utf-8')
        st.download_button(label="Download JSON File", data=json_data, file_name='cleaned_data.json', mime='application/json')

    st.success("Data Cleaning Completed!")
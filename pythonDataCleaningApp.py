import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import io  # Required for in-memory file handling

#Set Wide Layout
st.set_page_config(layout="wide")


# Title of the dashboard
st.title("Data Cleaning Dashboard")

# Initialize session state to store converted column descriptions
if 'converted_columns' not in st.session_state:
    st.session_state.converted_columns = {}

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

    # 1. Data Summary for Numerical Columns
    if st.checkbox("Show Data Summary"):
        st.write("### General Summary")
        st.write(f"Number of entries: {len(df)}")
        st.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.write("##Data Types")
        df_types = pd.DataFrame(df.dtypes).reset_index()
        df_types.columns = ['Column Name', 'Data Type']
        st.dataframe(df_types)

        # Display summary for numerical columns only
        st.write("### Summary Statistics for Numerical Columns:")
        
        summary_df = pd.DataFrame(df.describe()).reset_index()
        summary_df.columns = ['Statistics', 'Values']
        st.dataframe(summary_df)
        
        
        # Handling unsupported data types
        unsupported_columns = [col for col in df.columns if df[col].dtype in ['object', 'complex', 'mixed', 'timedelta64[ns]', 'sparse']]

        # Handle unsupported columns
        if unsupported_columns:
            st.warning(f"The following columns contain unsupported data types: {unsupported_columns}")
            col_to_convert = st.selectbox("Select a column to convert:", unsupported_columns)

            conversion_method = st.selectbox("Choose conversion method:", ["None", "To Numeric", "To String", "Drop Column"])

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
            st.write("### Summary Statistics for Converted Columns:")
            cols = st.columns(len(st.session_state.converted_columns))
            for i, (col_name, summary) in enumerate(st.session_state.converted_columns.items()):
                with cols[i]:
                    st.write(f"**{col_name}**")
                    
                    summary_df = pd.DataFrame(summary).reset_index()
                    summary_df.columns = ['Statistics', 'Values']  # Rename columns
                    st.dataframe(summary_df)

    # 2. Handling Missing Values
    if st.checkbox("Handle Missing Values"):
        st.write("### Missing Values Summary:")
        
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

    st.success("Data cleaning completed!")
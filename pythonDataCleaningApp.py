import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt
import xlsxwriter
import io  
import re
from thefuzz import fuzz, process 
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Set Wide Layout
st.set_page_config(layout="wide")

# Title of the dashboard
st.title("Data Cleaning Dashboard")

# Initialize session state to store converted column descriptions, DataFrame, and undo history
if 'converted_columns' not in st.session_state:
    st.session_state.converted_columns = {}
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'history' not in st.session_state:
    st.session_state.history = []  # To store history for undo

# File Upload
uploaded_file = st.file_uploader("Upload File for Data Cleaning", type=["csv", "xls", "xlsx", "json"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1]

    if file_type == 'csv':
        st.session_state.original_df = pd.read_csv(uploaded_file)
    elif file_type in ['xls', 'xlsx']:
        st.session_state.original_df = pd.read_excel(uploaded_file)
    elif file_type == 'json':
        st.session_state.original_df = pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file type!")

    # 1a. Automatically convert all text data to lowercase and sanitize
    def clean_and_convert_text_columns(df):
        text_columns = df.select_dtypes(include='object').columns
        for col in text_columns:
            # Convert text to lowercase, strip whitespace, and remove unwanted characters while keeping URL structure
            df[col] = df[col].str.lower().str.strip().apply(lambda x: re.sub(r'[^a-zA-Z0-9:/._?=&-]', ' ', str(x)))
        return df

    if 'df' not in st.session_state or st.session_state.df is None:
        st.session_state.df = clean_and_convert_text_columns(st.session_state.original_df.copy())
        st.success("Automatically Sanitized And Converted All Text Columns To Lowercase For Data Cleaning.")

    # 1b. Apply automatic data type detection for object columns
    st.session_state.df = st.session_state.df.infer_objects()
    st.success("Automatically Detected And Converted Suitable Object Columns To Appropriate Data Types.")

    # 2a. Data Summary for Columns
    st.write("### Showing Data Summary")
    if st.checkbox("Data Summary"):

        with st.expander("### Data Preview"):
            st.dataframe(st.session_state.df)

        st.write("### General Summary")
        st.write(f"Number of entries: {len(st.session_state.df)}")
        st.write(f"Memory Usage: {st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        with st.expander("### View Data Types"):
            df_types = pd.DataFrame(st.session_state.df.dtypes).reset_index()
            df_types.columns = ['Column Name', 'Data Type']
            st.dataframe(df_types)

        # 2b. Columns With Supported Data Types: numerical, strings, boolean, categorical, DateTime
        with st.expander("### Columns with Supported Data Types"):
            supported_columns = st.session_state.df.select_dtypes(include=['number','string', 'bool','category', 'datetime64']).columns.tolist()
            if supported_columns:
                st.write(supported_columns)
            else:
                st.warning("No Supported Columns Found In The Dataset!")

        # 2c. Columns with Unsupported Data Types (after applying infer_objects)
        with st.expander("### Columns with Unsupported Data Types"):
            unsupported_columns = [col for col in st.session_state.df.columns if st.session_state.df[col].dtype in ['object', 'complex', 'mixed', 'timedelta64[ns]', 'sparse']]
            st.warning("The Following Columns Contain Unsupported Data Types:")
            st.write(unsupported_columns)

            col_to_convert = st.selectbox("Select a Column to Convert to a Supported Data Type:", unsupported_columns)

            # 2ci. Conversion method now includes Categorical and DateTime
            conversion_method = st.selectbox("Choose a Supported Data Type:", ["To Numeric", "To String", "To Boolean", "To Categorical", "To DateTime", "None"])

            if st.button("Convert"):
                if col_to_convert:
                    # Save current dataframe state before applying conversion for undo functionality
                    st.session_state.history.append(st.session_state.df.copy())

                    # Perform the conversion
                    if conversion_method == "To Numeric":
                        st.session_state.df[col_to_convert] = pd.to_numeric(st.session_state.df[col_to_convert], errors='coerce')
                        st.success(f"Converted column '{col_to_convert}' to numeric.")
                    elif conversion_method == "To String":
                        st.session_state.df[col_to_convert] = st.session_state.df[col_to_convert].astype(str)
                        st.success(f"Converted column '{col_to_convert}' to string.")
                    elif conversion_method == "To Boolean":
                        st.session_state.df[col_to_convert] = st.session_state.df[col_to_convert].astype(bool)
                        st.success(f"Converted column '{col_to_convert}' to boolean.")
                    elif conversion_method == "To Categorical":
                        # Convert to categorical
                        st.session_state.df[col_to_convert] = st.session_state.df[col_to_convert].astype('category')
                        st.success(f"Converted column '{col_to_convert}' to categorical.")
                    elif conversion_method == "To DateTime":
                        st.session_state.df[col_to_convert] = pd.to_datetime(st.session_state.df[col_to_convert], errors='coerce')
                        st.success(f"Converted column '{col_to_convert}' to DateTime.")
                    else:
                        st.warning("Data Type Not Available For Conversion.")

                    # Store conversion history
                    st.session_state.converted_columns[col_to_convert] = conversion_method

            # 2cii. Undo button functionality
            if st.button("Undo"):
                if st.session_state.history:
                    st.session_state.df = st.session_state.history.pop()
                    st.success(f"Undo Conversion Successful! Of Column '{col_to_convert}'.")
                else:
                    st.warning("No more steps to undo!")

            # 2ciii. Display updated data types after conversion
            df_types = pd.DataFrame(st.session_state.df.dtypes).reset_index()
            df_types.columns = ['Column Name', 'Data Type']
            st.write("### Updated Data Types:")
            st.dataframe(df_types)

        # 2d. Display Conversion History
        st.write("### Conversion History")
        if st.session_state.converted_columns:
            history_df = pd.DataFrame(list(st.session_state.converted_columns.items()), columns=["Column Name", "Converted To"])
            st.dataframe(history_df)
        else:
            st.warning("No Columns Have Been Converted Yet!")

        # 2e. Display Summary For Dataset using df.describe() (using session state)
        st.write("### Summary Statistics for All Columns:")
        summary_df = pd.DataFrame(st.session_state.df.describe(include='all')).reset_index()
        summary_df.columns = ['Statistics'] + list(st.session_state.df.columns)
        st.dataframe(summary_df)



    # 2. Handling Missing Values
    st.write("### Handling Missing Values")
    if st.checkbox("Missing Values"):
        # Group 1: Mean, Median, Mode, Empty String, None
        with st.expander("### Basic Imputation"):
            missing_values_df = st.session_state.df.isnull().sum().add((st.session_state.df == '').sum()).add(st.session_state.df.isin([-9999, -1, 0, 'none', 'missing', 'na', 'not applicable']).sum()).reset_index()
            missing_values_df.columns = ['Column Name', 'Number of Missing Entries']
            st.dataframe(missing_values_df)

            columns_with_missing = missing_values_df[missing_values_df['Number of Missing Entries'] > 0]['Column Name'].tolist()
            selected_column_1 = st.selectbox("Select a column to handle missing values from Group 1", columns_with_missing)
            method_1 = st.selectbox("Fill missing values with:", ("None", "Mean", "Median", "Mode", "Empty String"))

            if st.button("Apply Basic Imputation"):
                if method_1 == "Mean":
                    st.session_state.df[selected_column_1].fillna(st.session_state.df[selected_column_1].mean(), inplace=True)
                elif method_1 == "Median":
                    st.session_state.df[selected_column_1].fillna(st.session_state.df[selected_column_1].median(), inplace=True)
                elif method_1 == "Mode":
                    st.session_state.df[selected_column_1].fillna(st.session_state.df[selected_column_1].mode()[0], inplace=True)
                elif method_1 == "Empty String":
                    st.session_state.df[selected_column_1].fillna('', inplace=True)

                st.success(f"Filled missing values in '{selected_column_1}' using '{method_1}'.")
                # Display the final DataFrame after handling missing values
                st.dataframe(st.session_state.df)

        # Group 2: Forward Fill, Backward Fill
        with st.expander("### Forward and Backward Fill"):
            missing_values_df = st.session_state.df.isnull().sum().add((st.session_state.df == '').sum()).add(st.session_state.df.isin([-9999, -1, 0, 'none', 'missing', 'na', 'not applicable']).sum()).reset_index()
            missing_values_df.columns = ['Column Name', 'Number of Missing Entries']
            st.dataframe(missing_values_df)

            selected_column_2 = st.selectbox("Select a column for Group 2 Fill", columns_with_missing)
            method_2 = st.selectbox("Fill missing values with:", ("Forward Fill", "Backward Fill"))

            if st.button("Apply Foward/Backward Filling"):
                if method_2 == "Forward Fill":
                    st.session_state.df[selected_column_2].fillna(method='ffill', inplace=True)
                elif method_2 == "Backward Fill":
                    st.session_state.df[selected_column_2].fillna(method='bfill', inplace=True)

                st.success(f"Applied '{method_2}' to column '{selected_column_2}'.")
                # Display the final DataFrame after handling missing values
                st.dataframe(st.session_state.df)


        # Group 3: Imputation using ML models
        with st.expander("### Machine Learning Imputation"):
            missing_values_df = st.session_state.df.isnull().sum().add((st.session_state.df == '').sum()).add(st.session_state.df.isin([-9999, -1, 0, 'none', 'missing', 'na', 'not applicable']).sum()).reset_index()
            missing_values_df.columns = ['Column Name', 'Number of Missing Entries']
            st.dataframe(missing_values_df)

            selected_column_3 = st.selectbox("Select a column for Group 3 Imputation", columns_with_missing)
            method_3 = st.selectbox("Impute missing values with:", ("KNN", "Random Forest", "Linear Regression"))

            if st.button("Apply Imputation"):
                features = st.session_state.df.drop(columns=[selected_column_3]).select_dtypes(include=np.number)
                target = st.session_state.df[selected_column_3]

                # Handle missing values in the feature set (simplified for demo purposes)
                features.fillna(features.mean(), inplace=True)

                if method_3 == "KNN":
                    knn_imputer = KNNImputer(n_neighbors=5)
                    st.session_state.df[selected_column_3] = knn_imputer.fit_transform(st.session_state.df[[selected_column_3]])

                elif method_3 == "Random Forest":
                    rf = RandomForestRegressor(n_estimators=100)
                    rf.fit(features, target.fillna(target.median()))  # Train on non-missing values
                    missing_mask = target.isnull()
                    st.session_state.df.loc[missing_mask, selected_column_3] = rf.predict(features[missing_mask])

                elif method_3 == "Linear Regression":
                    lr = LinearRegression()
                    lr.fit(features, target.fillna(target.median()))  # Train on non-missing values
                    missing_mask = target.isnull()
                    st.session_state.df.loc[missing_mask, selected_column_3] = lr.predict(features[missing_mask])

                st.success(f"Imputed missing values in '{selected_column_3}' using '{method_3}'.")
                # Display the final DataFrame after handling missing values
                st.dataframe(st.session_state.df)


        # Group 4: Remove rows or columns with missing values
        with st.expander("### Remove Missing Data"):
            missing_values_df = st.session_state.df.isnull().sum().add((st.session_state.df == '').sum()).add(st.session_state.df.isin([-9999, -1, 0, 'none', 'missing', 'na', 'not applicable']).sum()).reset_index()
            missing_values_df.columns = ['Column Name', 'Number of Missing Entries']
            st.dataframe(missing_values_df)

            method_4 = st.selectbox("Remove missing data by:", ("Remove Rows", "Remove Columns"))

            if st.button("Remove Missing Data"):
                if method_4 == "Remove Rows":
                    st.session_state.df.dropna(inplace=True)
                elif method_4 == "Remove Columns":
                    st.session_state.df.dropna(axis=1, inplace=True)

                st.success(f"Applied '{method_4}' to handle missing data.")
                # Display the final DataFrame after handling missing values
                st.dataframe(st.session_state.df)
            
            
        
    # 3. Handling Duplicates
    st.write("### Handling Duplicates")
    if st.checkbox("Duplicates"):
        # a) Identifying Duplicates
        st.write("##### Identifying Duplicates")
        
        # i) Exact Matches
        if st.checkbox("Identifying Exact Duplicates"):
            exact_dup = st.session_state.df[st.session_state.df.duplicated(keep=False)]
            st.write(f"**Exact Duplicates**: {exact_dup.shape[0]} entries found")
            with st.expander("Showing Exact Duplicates"):
                st.dataframe(exact_dup)

        # ii) Fuzzy Matching
        if st.checkbox("Identifying Almost Similar Duplicates"):
            # Select column for fuzzy matching
            column_to_match = st.selectbox("Select column for fuzzy matching", st.session_state.df.columns)
            threshold = st.slider("Set similarity threshold for fuzzy matching", 70, 100, 90)
            matches = []
            for i, val in enumerate(st.session_state.df[column_to_match]):
                for j in range(i + 1, len(st.session_state.df)):
                    score = fuzz.ratio(val, st.session_state.df[column_to_match].iloc[j])
                    if score >= threshold:
                        matches.append((i, j, val, st.session_state.df[column_to_match].iloc[j], score))

            fuzzy_dup_df = pd.DataFrame(matches, columns=["Row 1", "Row 2", "Value 1", "Value 2", "Score"])
            st.write(f"**Fuzzy Matches**: {fuzzy_dup_df.shape[0]} pairs found")
            with st.expander("Showing Fuzzy Matches"):
                st.dataframe(fuzzy_dup_df)
                
        # b) Handling Duplicates
        st.write("##### Removing Duplicates")

        # i) Removing Duplicates
        if st.checkbox("Removing Exact Duplicates"):
            st.session_state.df.drop_duplicates(inplace=True)
            st.success("Exact duplicates removed.")
            with st.expander("Data After Removing Duplicates"):
                st.dataframe(st.session_state.df)
                
        # ii) Aggregating Data
        if st.checkbox("Aggregating Duplicates"):
            agg_column = st.selectbox("Select column to aggregate duplicate data", st.session_state.df.columns)
            agg_method = st.selectbox("Choose aggregation method", ["sum", "mean", "count", "first", "last"])
            
            if agg_method == "sum":
                df_agg = st.session_state.df.groupby(agg_column).sum()
            elif agg_method == "mean":
                df_agg = st.session_state.df.groupby(agg_column).mean()
            elif agg_method == "count":
                df_agg = st.session_state.df.groupby(agg_column).count()
            elif agg_method == "first":
                df_agg = st.session_state.df.groupby(agg_column).first()
            elif agg_method == "last":
                df_agg = st.session_state.df.groupby(agg_column).last()
            
            st.success(f"Data aggregated using '{agg_method}' method.")
            with st.expander("Aggregated Data"):
                st.dataframe(df_agg)


    # 4. Outlier Detection
    st.write("### Handling Outliers")
    if st.checkbox("Outlier Detection"):
        numerical_columns = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        col = st.selectbox("Select column for outlier detection", numerical_columns)
        if col:
            Q1 = st.session_state.df[col].quantile(0.25)
            Q3 = st.session_state.df[col].quantile(0.75)
            IQR = Q3 - Q1
            st.session_state.df = st.session_state.df[~((st.session_state.df[col] < (Q1 - 1.5 * IQR)) | (st.session_state.df[col] > (Q3 + 1.5 * IQR)))]
            st.write(f"Outliers removed based on IQR for column {col}")
            st.dataframe(st.session_state.df)


    # 5. Normalization and Scaling
    st.write("### Normalizing Data")
    if st.checkbox("Normalization"):
        method = st.selectbox("Choose normalization method:", ["None", "Z-score", "MinMax", "Robust"])
        columns = st.multiselect("Select columns to normalize/scale", st.session_state.df.select_dtypes(include=np.number).columns)

        if method == "Z-score":
            st.session_state.df[columns] = StandardScaler().fit_transform(st.session_state.df[columns])
        elif method == "MinMax":
            st.session_state.df[columns] = MinMaxScaler().fit_transform(st.session_state.df[columns])
        elif method == "Robust":
            st.session_state.df[columns] = RobustScaler().fit_transform(st.session_state.df[columns])

        st.write("### Data After Scaling")
        st.dataframe(st.session_state.df)


    # 6. Text Data Consistency
    st.write("### Handling Text Data Consistency")
    if st.checkbox("Text Data Consistency"):
        st.write("Automatically converting all text columns to lowercase unless altered here.")
        
        # Provide option to change case
        col = st.selectbox("Select text column to modify:", st.session_state.df.select_dtypes(include='object').columns)
        text_case = st.selectbox("Convert text case:", ["Lowercase", "Uppercase", "Titlecase"])

        if text_case == "Lowercase":
            st.session_state.df[col] = st.session_state.df[col].str.lower()
        elif text_case == "Uppercase":
            st.session_state.df[col] = st.session_state.df[col].str.upper()
        elif text_case == "Titlecase":
            st.session_state.df[col] = st.session_state.df[col].str.title()

        st.success(f"Text in column '{col}' converted to {text_case}.")
        st.dataframe(st.session_state.df)


    # 7. Data Export
    st.write("### Export Cleaned Data")
    if st.checkbox("Data Export"):
        # User can select the export format (Excel, CSV, JSON)
        export_format = st.selectbox("Choose export format:", ["Excel", "CSV", "JSON"])

        # Variable to track if the button is clicked
        download_clicked = False

        # Exporting as Excel
        if export_format == "Excel":
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                st.session_state.df.to_excel(writer, index=False) 
                buffer.seek(0) 
            download_clicked = st.download_button(
                label="Download Cleaned Data as Excel",
                data=buffer,
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Exporting as CSV
        elif export_format == "CSV":
            csv = st.session_state.df.to_csv(index=False).encode('utf-8')
            download_clicked = st.download_button(
                label="Download Cleaned Data as CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
        

        # Exporting as JSON
        elif export_format == "JSON":
            json_data = st.session_state.df.to_json(orient="records")
            download_clicked = st.download_button(
                label="Download Cleaned Data as JSON",
                data=json_data,
                file_name="cleaned_data.json",
                mime="application/json"
            )

        # Show success message after download button is clicked
        if download_clicked:
            st.success("The cleaned dataset has been successfully downloaded!")
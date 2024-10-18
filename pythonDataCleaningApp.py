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

    # 1a. Automatically convert text columns to lowercase and sanitize
    def clean_and_convert_text_columns(df):
        text_columns = df.select_dtypes(include='object').columns
        for col in text_columns:
            df[col] = df[col].str.lower().str.strip().apply(lambda x: re.sub(r'[^a-zA-Z0-9:/._?=&-]', ' ', str(x)))
        return df

    # Create a fresh copy of the original DataFrame
    if 'df' not in st.session_state or st.session_state.df is None:
        st.session_state.df = clean_and_convert_text_columns(st.session_state.original_df.copy())
        st.success("Sanitized and converted text columns to lowercase.")

    # Apply automatic data type detection for object columns
    st.session_state.df = st.session_state.df.infer_objects()
    st.success("Detected and converted suitable object columns to appropriate data types.")

    # 1b. Display Data Summary
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

        # 1bi. Columns with Supported Data Types
        with st.expander("### Columns with Supported Data Types"):
            supported_columns = st.session_state.df.select_dtypes(include=['number','string', 'bool','category', 'datetime64']).columns.tolist()
            if supported_columns:
                st.write(supported_columns)
            else:
                st.warning("No supported columns found in the dataset!")

        # 1bii. Columns with Unsupported Data Types (after applying infer_objects)
        with st.expander("### Columns with Unsupported Data Types"):
            unsupported_columns = [col for col in st.session_state.df.columns if st.session_state.df[col].dtype in ['object','category','sparse', 'complex', 'mixed', 'timedelta64[ns]']]
            st.warning("The following columns contain unsupported data types:")
            st.write(unsupported_columns)

            col_to_convert = st.selectbox("Select a Column to Convert to a Supported Data Type:", unsupported_columns)

            # Conversion method now includes DateTime
            conversion_method = st.selectbox("Choose a Supported Data Type:", ["To Numeric", "To String", "To Boolean", "To DateTime", "None"])

            if st.button("Convert"):
                if col_to_convert:
                    # Save current dataframe state before applying conversion for undo functionality
                    st.session_state.history.append(st.session_state.df.copy())

                    # Perform the conversion
                    if conversion_method == "To Numeric":
                        st.session_state.df[col_to_convert] = pd.to_numeric(st.session_state.df[col_to_convert], errors='coerce')
                        st.success(f"Converted column '{col_to_convert}' to numeric.")
                    elif conversion_method == "To String":
                        st.session_state.df[col_to_convert] = st.session_state.df[col_to_convert].astype("string").replace({'none': np.nan, 'nan': np.nan})
                        st.success(f"Converted column '{col_to_convert}' to string.")
                    elif conversion_method == "To Boolean":
                        st.session_state.df[col_to_convert] = st.session_state.df[col_to_convert].astype(bool)
                        st.success(f"Converted column '{col_to_convert}' to boolean.")
                    elif conversion_method == "To DateTime":
                        st.session_state.df[col_to_convert] = pd.to_datetime(st.session_state.df[col_to_convert], errors='coerce')
                        st.success(f"Converted column '{col_to_convert}' to DateTime.")
                    else:
                        st.warning("Data Type not available for conversion.")

                    # Store conversion history
                    st.session_state.converted_columns[col_to_convert] = conversion_method

            # Undo button functionality
            if st.button("Undo Data Type Conversion"):
                if st.session_state.history:
                    st.session_state.df = st.session_state.history.pop()
                    st.success(f"Undo Data Type Conversion For Column '{col_to_convert}'.")
                else:
                    st.warning("No more steps to undo!")

            # Display updated data types after conversion
            df_types = pd.DataFrame(st.session_state.df.dtypes).reset_index()
            df_types.columns = ['Column Name', 'Data Type']
            st.write("### Updated Data Types:")
            st.dataframe(df_types)
            

        # 1c. Display Conversion History
        st.write("### Conversion History")
        if st.session_state.converted_columns:
            history_df = pd.DataFrame(list(st.session_state.converted_columns.items()), columns=["Column Name", "Converted To"])
            st.dataframe(history_df)
        else:
            st.warning("No columns have been converted yet!")

        # 1d. Summary Statistics for all columns using df.describe()
        st.write("### Summary Statistics for All Columns:")
        summary_df = pd.DataFrame(st.session_state.df.describe(include='all')).reset_index()
        summary_df.columns = ['Statistics'] + list(st.session_state.df.columns)
        st.dataframe(summary_df)



    # 2a. Handling Missing Values
    st.write("### Handling Missing Values")
    if st.checkbox("Missing Values"):
        # Recalculate missing values before handling
        def update_missing_values_df():
            missing_values_df = st.session_state.df.isnull().sum().add(st.session_state.df.applymap(lambda x: isinstance(x, str) and x.strip().lower() in ['', 'none', 'missing', 'na', 'not applicable', 'null']).sum()).add(st.session_state.df.isin([-9999, -999, -1, 0, 999, 9999, np.inf, -np.inf]).sum()).reset_index()
            missing_values_df.columns = ['Column Name', 'Number Of Missing Entries']
            return missing_values_df
        
        # 2b. Basic Imputation: Mean, Median, Mode, Empty String, None
        with st.expander("### Basic Imputation"):
            #Show Missing Values
            missing_values_df = update_missing_values_df()
            st.dataframe(missing_values_df)
            
            columns_with_missing = missing_values_df[missing_values_df['Number Of Missing Entries'] > 0]['Column Name'].tolist()
            selected_column_1 = st.selectbox("Select A Column To handle Missing Values:", columns_with_missing)
            method_1 = st.selectbox("Fill Missing Values With:", ("None", "Mean", "Median", "Mode", "Empty String"))

            if st.button("Apply Basic Imputation"):
                
                # Save the current state before applying the removal for undo functionality
                st.session_state.history.append(st.session_state.df.copy())
                
                if method_1 == "Mean":
                    st.session_state.df[selected_column_1].fillna(st.session_state.df[selected_column_1].mean(), inplace=True)
                elif method_1 == "Median":
                    st.session_state.df[selected_column_1].fillna(st.session_state.df[selected_column_1].median(), inplace=True)
                elif method_1 == "Mode":
                    st.session_state.df[selected_column_1].fillna(st.session_state.df[selected_column_1].mode()[0], inplace=True)
                elif method_1 == "Empty String":
                    st.session_state.df[selected_column_1].fillna('', inplace=True)

                st.success(f"Filled Missing Values In '{selected_column_1}' Using '{method_1}'.")
                st.dataframe(st.session_state.df)
            
            # Undo button functionality
            if st.button("Undo Basic Imputation"):
                if st.session_state.history:
                    st.session_state.df = st.session_state.history.pop()
                    st.success(f"Undo Basic Imputation For Column '{selected_column_1}'.")
                else:
                    st.warning("No More Steps To Undo!")


        # 2c. Forward Fill and Backward Fill
        with st.expander("### Forward Fill and Backward Fill"):
            #Show Missing Values
            missing_values_df = update_missing_values_df()
            st.dataframe(missing_values_df)
            
            selected_column_2 = st.selectbox("Select a column to handle missing values (Forward/Backward Fill):", columns_with_missing)
            method_2 = st.selectbox("Fill missing values with:", ("Forward Fill", "Backward Fill"))

            if st.button("Apply Forward/Backward Fill"):
                # Save the current state before applying the fill for undo functionality
                st.session_state.history.append(st.session_state.df.copy())

                if method_2 == "Forward Fill":
                    st.session_state.df[selected_column_2].fillna(method='ffill', inplace=True)
                    st.success(f"Applied Forward Fill to '{selected_column_2}'.")
                elif method_2 == "Backward Fill":
                    st.session_state.df[selected_column_2].fillna(method='bfill', inplace=True)
                    st.success(f"Applied Backward Fill to '{selected_column_2}'.")

                st.dataframe(st.session_state.df)
                
            # Undo button functionality
            if st.button("Undo Forward/Backward Fill"):
                if st.session_state.history:
                    st.session_state.df = st.session_state.history.pop()
                    st.success(f"Undo Forward/ Backward Fill For Column '{selected_column_2}'.")
                else:
                    st.warning("No more steps to undo!")
            

        # 2d. Imputation Using Machine Learning Models
        with st.expander("### Imputation Using Machine Learning Models"):
            #Show Missing Values
            missing_values_df = update_missing_values_df()
            st.dataframe(missing_values_df)
            
            selected_column_3 = st.selectbox("Select A Column To Handle Missing Values (ML Imputation):", columns_with_missing)
            ml_method = st.selectbox("Impute using:", ("KNN", "Random Forest", "Linear Regression"))

            # Select feature columns to use for the imputation
            feature_columns = st.multiselect("Select feature columns to use for imputation:", st.session_state.df.columns.tolist())

            if st.button("Apply ML Imputation"):
                if feature_columns and selected_column_3:
                    # Save the current state before applying the ML imputation for undo functionality
                    st.session_state.history.append(st.session_state.df.copy())

                    df_ml = st.session_state.df[feature_columns + [selected_column_3]].copy()
                    
                    # Ensure there are no missing values in the feature columns
                    df_ml.dropna(subset=feature_columns, inplace=True)

                    if ml_method == "KNN":
                        knn_imputer = KNNImputer()
                        df_imputed = pd.DataFrame(knn_imputer.fit_transform(df_ml), columns=df_ml.columns)
                        st.session_state.df[selected_column_3] = df_imputed[selected_column_3]
                        st.success(f"Applied KNN Imputation to '{selected_column_3}'.")
                    
                    elif ml_method == "Random Forest":
                        # Train the Random Forest model on rows without missing values in the selected column
                        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        non_missing_df = df_ml[df_ml[selected_column_3].notnull()]
                        missing_df = df_ml[df_ml[selected_column_3].isnull()]

                        rf_model.fit(non_missing_df[feature_columns], non_missing_df[selected_column_3])
                        imputed_values = rf_model.predict(missing_df[feature_columns])

                        st.session_state.df.loc[st.session_state.df[selected_column_3].isnull(), selected_column_3] = imputed_values
                        st.success(f"Applied Random Forest Imputation to '{selected_column_3}'.")

                    elif ml_method == "Linear Regression":
                        # Train the Linear Regression model on rows without missing values in the selected column
                        lr_model = LinearRegression()
                        non_missing_df = df_ml[df_ml[selected_column_3].notnull()]
                        missing_df = df_ml[df_ml[selected_column_3].isnull()]

                        lr_model.fit(non_missing_df[feature_columns], non_missing_df[selected_column_3])
                        imputed_values = lr_model.predict(missing_df[feature_columns])

                        st.session_state.df.loc[st.session_state.df[selected_column_3].isnull(), selected_column_3] = imputed_values
                        st.success(f"Applied Linear Regression Imputation to '{selected_column_3}'.")

                else:
                    st.warning("Please select at least one feature column for imputation.")

                st.dataframe(st.session_state.df)
                
            # Undo button functionality
            if st.button("Undo ML Imputation"):
                if st.session_state.history:
                    st.session_state.df = st.session_state.history.pop()
                    st.success(f"Undo ML Imputation For Column '{selected_column_3}'.")
                else:
                    st.warning("No more steps to undo!")
             

        # 2e. Remove Rows or Columns with Missing Values
        with st.expander("### Remove Rows Or Columns with Missing Values"):
            #Show Missing Values
            missing_values_df = update_missing_values_df()
            st.dataframe(missing_values_df)
            
            removal_method = st.selectbox("Remove Rows Or Columns:", ("Removing Rows With Missing Values", "Removing Columns with Missing Values"))

            if st.button("Apply Removal"):
                # Save the current state before applying the removal for undo functionality
                st.session_state.history.append(st.session_state.df.copy())

                if removal_method == "Remove Rows with Missing Values":
                    st.session_state.df.dropna(inplace=True)
                    st.success("Removed rows with missing values.")
                elif removal_method == "Remove Columns with Missing Values":
                    st.session_state.df.dropna(axis=1, inplace=True)
                    st.success("Removed columns with missing values.")

                st.dataframe(st.session_state.df)
                
            # Undo button functionality
            if st.button("Undo Column/Row Removal"):
                if st.session_state.history:
                    st.session_state.df = st.session_state.history.pop()
                    st.success(f"Undo Successful For '{removal_method}'.")
                else:
                    st.warning("No More Steps To Undo!")
                    
            
        # Update missing values after handling
        with st.expander("### Preview Of DataSet After Handling Missing Values"):
            missing_values_df = update_missing_values_df()
            st.dataframe(missing_values_df)


        
    # 3. Handling Duplicates
    st.write("### Handling Duplicates")
    if st.checkbox("Duplicates"):
        # a) Identifying Duplicates
        st.write("##### Identifying Duplicates")
        
        # i) Exact Matches
        with st.expander("Identifying Exact Duplicates"):
            exact_dup = st.session_state.df[st.session_state.df.duplicated(keep=False)]
            st.write(f"**Exact Duplicates**: {exact_dup.shape[0]} entries found")
            st.write("Showing Exact Duplicates")
            st.dataframe(exact_dup)


        # ii) Fuzzy Matching
        with st.expander("Identifying Almost Similar Duplicates"):
            # Select column for fuzzy matching
            string_columns = st.session_state.df.select_dtypes(include=["object", "string"]).columns
            
            if len(string_columns) == 0:
                st.warning("No string columns available for fuzzy matching.")
            else:
                column_to_match = st.selectbox("Select column for fuzzy matching", string_columns)

                # Store the original data types of the entire DataFrame
                original_dtypes = st.session_state.df.dtypes

                # Convert selected column to string for fuzzy matching
                st.session_state.df[column_to_match] = st.session_state.df[column_to_match].astype(str)

                # Slider to set the similarity threshold
                threshold = st.slider("Set similarity threshold for fuzzy matching", 70, 100, 90)

                # Perform fuzzy matching only if a string-based column is selected
                matches = []
                for i, val in enumerate(st.session_state.df[column_to_match]):
                    for j in range(i + 1, len(st.session_state.df)):
                        # Perform fuzzy matching only if both values are strings
                        val_2 = st.session_state.df[column_to_match].iloc[j]
                        if pd.notnull(val) and pd.notnull(val_2):  # Ensure no NaN values
                            score = fuzz.ratio(val, val_2)  # Perform fuzzy matching on strings
                            if score >= threshold:
                                matches.append((i, j, val, val_2, score))

                # Create a DataFrame with the matches
                fuzzy_dup_df = pd.DataFrame(matches, columns=["Row 1", "Row 2", "Value 1", "Value 2", "Score"])
                
                st.write(f"**Fuzzy Matches**: {fuzzy_dup_df.shape[0]} pairs found")
                st.write("Showing Fuzzy Matches")
                st.dataframe(fuzzy_dup_df)

                # Restore the original data types after fuzzy matching is done
                st.session_state.df = st.session_state.df.astype(original_dtypes)

                # Show confirmation that data types have been restored
                st.write("Original data types have been restored.")
                st.dataframe(pd.DataFrame(st.session_state.df.dtypes, columns=["Data Type"]))

                        
        # b) Handling Duplicates
        st.write("##### Removing Duplicates")

        # i) Removing Duplicates
        with st.expander("Removing Exact Duplicates"):
            # Save the current state before applying the removal for undo functionality
            st.session_state.history.append(st.session_state.df.copy())
            
            
            st.session_state.df.drop_duplicates(inplace=True)
            st.success("Exact duplicates removed.")
            st.write("Data After Removing Duplicates")
            st.dataframe(st.session_state.df)
                
        # Undo button functionality
        if st.button("Undo Duplicate Removal"):
            if st.session_state.history:
                st.session_state.df = st.session_state.history.pop()
                st.success("Undo Duplicate Removal.")
            else:
                st.warning("No more steps to undo!")
                
    
                
        # ii) Aggregating Data
        with st.expander("Aggregating Duplicates"):
            # Save the current state before applying the removal for undo functionality
            st.session_state.history.append(st.session_state.df.copy())
            
            agg_column = st.selectbox("Select column to Aggregate Duplicate Data", st.session_state.df.columns)
            agg_method = st.selectbox("Choose Aggregation Method", ["Sum", "Mean", "Count", "First", "Last"])
            
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
            else:
                df_agg = None #Fallback to None if the method does not match
            
            #Only Display The Aggregated DataFrames If It Was Defined
            if df_agg is not None:
                st.success(f"Data aggregated using '{agg_method}' method.")
                st.write("Aggregated Data")
                st.dataframe(df_agg)
            else:
                st.warning("No Valid ggregation Method Selection.")
                    
        # Undo button functionality
        if st.button("Undo Data Aggregation"):
                if st.session_state.history:
                    st.session_state.df = st.session_state.history.pop()
                    st.success(f"Undo Successful Aggregation For Column '{agg_column}'.")
                else:
                    st.warning("No More Steps To Undo!")
                    
    
    # 4a. Outlier Detection
    st.write("### Handling Outliers")
    if st.checkbox("Outlier Detection"):
        with st.expander("IQR"):
            numerical_columns = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
            col = st.selectbox("Select Column For Outlier Detection", numerical_columns)
            
            if st.button("Apply IQR"):
                # Save the current state before applying the removal for undo functionality
                st.session_state.history.append(st.session_state.df.copy())
                if col:
                    Q1 = st.session_state.df[col].quantile(0.25)
                    Q3 = st.session_state.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    st.session_state.df = st.session_state.df[~((st.session_state.df[col] < (Q1 - 1.5 * IQR)) | (st.session_state.df[col] > (Q3 + 1.5 * IQR)))]
                    st.success(f"Successfully Removed Outliers Based On IQR For Column {col}")
                st.dataframe(st.session_state.df)
            
            # Undo button functionality
            if st.button("Undo Outlier Removal"):
                if st.session_state.history:
                    st.session_state.df = st.session_state.history.pop()
                    st.success(f"Undo Successful Outlier Detection For Column '{col}'.")
                else:
                    st.warning("No More Steps To Undo!")
                    
           
    # 5a. Normalization and Scaling
    st.write("### Normalizing Data")
    if st.checkbox("Normalization"):
        with st.expander("Normalization & Scaling Data"):
            method = st.selectbox("Choose Normalization Method:", ["None", "Z-score", "MinMax", "Robust"])
            columns = st.multiselect("Select Columns To Normalize/Scale", st.session_state.df.select_dtypes(include=np.number).columns)
            if st.button("Apply Normalization"):
                # Save the current state before applying the removal for undo functionality
                st.session_state.history.append(st.session_state.df.copy())
            
                if method == "Z-score":
                    st.session_state.df[columns] = StandardScaler().fit_transform(st.session_state.df[columns])
                elif method == "MinMax":
                    st.session_state.df[columns] = MinMaxScaler().fit_transform(st.session_state.df[columns])
                elif method == "Robust":
                    st.session_state.df[columns] = RobustScaler().fit_transform(st.session_state.df[columns])

                st.success("### Successfully Normalized Data!")
            st.dataframe(st.session_state.df)
                
            # Undo button functionality
            if st.button("Undo Data Normalization"):
                if st.session_state.history:
                    st.session_state.df = st.session_state.history.pop()
                    st.success(f"Undo Normalization For Column/s '{columns}'.")
                else:
                    st.warning("No More Steps To Undo!")
                    
           


    # 6a. Text Data Consistency
    st.write("### Handling Text Data Consistency")
    if st.checkbox("Text Data Consistency"):
        with st.expander("Text Consistency"):     
            st.write("Automatically Converting All Text Columns To Lowercase Unless Altered Here.")
            # Provide option to change case
            col = st.selectbox("Select Text Column To Modify:", st.session_state.df.select_dtypes(include='object').columns)
            text_case = st.selectbox("Convert Text Case:", ["Lowercase", "Uppercase", "Titlecase"])
            
            if st.button("Apply Text Case"):
                # Save the current state before applying the removal for undo functionality
                st.session_state.history.append(st.session_state.df.copy())
                
                if text_case == "Lowercase":
                    st.session_state.df[col] = st.session_state.df[col].str.lower()
                elif text_case == "Uppercase":
                    st.session_state.df[col] = st.session_state.df[col].str.upper()
                elif text_case == "Titlecase":
                    st.session_state.df[col] = st.session_state.df[col].str.title()

                st.success(f"Successfully Converted Text In Column '{col}' To {text_case}.")
            st.dataframe(st.session_state.df)
            
            # Undo button functionality
            if st.button("Undo Text Casing"):
                if st.session_state.history:
                    st.session_state.df = st.session_state.history.pop()
                    st.success(f"Undo Text Casing For Column '{col}'.")
                else:
                    st.warning("No More Steps To Undo!")
                    
        

    # 7a. Data Export
    st.write("### Export Cleaned Data")
    if st.checkbox("Data Export"):
        with st.expander("Download Cleaned Data"):
            # User can select the export format (Excel, CSV, JSON)
            export_format = st.selectbox("Choose File Export Format:", ["Excel", "CSV", "JSON"])

            # Variable to track if the button is clicked
            download_clicked = False
            
        
            # i) Exporting as Excel
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
            
            # ii) Exporting as CSV
            elif export_format == "CSV":
                csv = st.session_state.df.to_csv(index=False).encode('utf-8')
                download_clicked = st.download_button(
                    label="Download Cleaned Data as CSV",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
            

            # iii) Exporting as JSON
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
                st.success("The Cleaned Dataset Has Been Successfully Downloaded!")
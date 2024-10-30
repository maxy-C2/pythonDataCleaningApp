import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from scipy.stats import skew, kurtosis, boxcox
import matplotlib.pyplot as plt
import xlsxwriter
import io  
import re
from thefuzz import fuzz
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

    def dataSummary():
        # 1a. Automatically convert text columns to lowercase and sanitize
        def clean_and_convert_text_columns(df):
            text_columns = df.select_dtypes(include='object').columns
            for col in text_columns:
                df[col] = df[col].str.lower().str.strip().apply(lambda x: re.sub(r'[^a-zA-Z0-9:/._?=&-]', ' ', str(x)))
            return df

        # Create a fresh copy of the original DataFrame
        if 'df' not in st.session_state or st.session_state.df is None:
            st.session_state.df = clean_and_convert_text_columns(st.session_state.original_df.copy())
            st.success("Successfully Sanitized and converted text columns to lowercase.")

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
                supported_columns = st.session_state.df.select_dtypes(include=['number','string', 'bool', 'datetime64']).columns.tolist()
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
            
    dataSummary()

    # Function for handling missing values with various techniques
    def handleMissingValues():
        # Helper function to update missing values DataFrame
        def update_missing_values_df():
            # Comprehensive handling of missing values
            missing_values_df = st.session_state.df.isnull().sum() \
                .add(st.session_state.df.map(lambda x: isinstance(x, str) and x.strip().lower() in ['', 'none', 'missing', 'na', 'not applicable', 'null']).sum()) \
                .add(st.session_state.df.isin([-9999, -999, 999, 9999, np.inf, -np.inf, np.nan]).sum()) \
                .reset_index() #removed -1 and 0 from the missing values
            missing_values_df.columns = ['Column Name', 'Number Of Missing Entries']
            return missing_values_df

        # Helper function to ensure a column is numeric, catch exceptions for non-numeric data        
        def ensure_numeric(df, column):
            try:
                if pd.api.types.is_numeric_dtype(df[column]):
                    return df[column]
            except Exception as e:
                st.error(f"The Column '{column}' has an incompatible data type for the selected technique to handle missing values: {e}")
            return df[column]  # Return even if failed, so handling continues

        # Helper function to apply imputation to each missing value in a column
        def impute_values(df, column, method):
            try:
                if method == "Mean":
                    df[column] = ensure_numeric(df, column)
                    df[column].fillna(df[column].mean(), inplace=True)
                elif method == "Median":
                    df[column] = ensure_numeric(df, column)
                    df[column].fillna(df[column].median(), inplace=True)
                elif method == "Mode":
                    df[column].fillna(df[column].mode()[0], inplace=True)
                elif method == "None":
                    df[column].fillna('None', inplace=True)
                
                # Extend handling for placeholders
                df[column].replace(['', 'none', 'missing', 'na', 'not applicable', 'null', -9999, -999, 999, 9999, np.inf, -np.inf, np.nan], np.nan, inplace=True)
            except Exception as e:
                st.error(f"Error applying {method} imputation on column '{column}': {e}")
            return df

        def create_missing_indicator_df():
            # Get the original DataFrame from the session state
            df = st.session_state.df

            # Create a binary DataFrame indicating missing values
            missing_indicator_df = pd.DataFrame()

            # Loop through each column to identify missing values
            for column in df.columns:
                missing_indicator_df[column] = df[column].isnull() | \
                    df[column].astype(str).str.strip().str.lower().isin(['', 'none', 'missing', 'na', 'not applicable', 'null']) | \
                    df[column].isin([-9999, -999, 999, 9999, np.inf, -np.inf, np.nan]) #removed 0 and -1 as missing values

            # Convert boolean values to integers (1 for missing, 0 for non-missing)
            missing_indicator_df = missing_indicator_df.astype(int)

            return missing_indicator_df

        st.write("### Handling Missing Values")

        if st.checkbox("Missing Values"):
            with st.expander("### Missing Values Summary Visualization."):
                selected_visualization_method = st.selectbox("Select Missing Values Visualization Method:", ["Heatmap", "Table"])
                missing_values_df = update_missing_values_df()
                missing_indicator_df = create_missing_indicator_df()
                
                if st.button("Show Missing Values Summary"):
                    if selected_visualization_method == "Heatmap":
                        # Configure plot size and aesthetics with a larger height-to-width ratio
                        plt.figure(figsize=(8, 12))  # Adjusted height for narrower columns

                        # Create a color map for missing (1) and non-missing (0) values
                        color_map = np.where(missing_indicator_df == 1, 1, 0)  # Missing values as 1, non-missing as 0

                        # Apply the color map with a custom palette
                        cmap = sns.color_palette("viridis", as_cmap=True)

                        # Draw the heatmap using the mask based on color_map
                        sns.heatmap(
                            missing_indicator_df,     # Data for the heatmap
                            cmap=cmap,                # Use the colormap
                            annot=False,               # Show annotations
                            #fmt='d',                  # Format annotations as integers
                            cbar_kws={'label': 'Missing Value Indicator'},  # Color bar label
                            linewidths=1,             # Line width between cells
                            linecolor='gray',         # Line color
                            #mask=(color_map == 0)     # Mask non-missing values
                        )

                        # Add title and format
                        plt.title('Heatmap of Missing Values by Column', fontsize=16, weight='bold')
                        plt.xlabel('Columns')
                        plt.ylabel('Rows')
                        plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels for readability
                        plt.yticks([])                                   # Hide y-axis labels to focus on columns
                        
                        # Adjust layout to make space for x-axis labels
                        plt.subplots_adjust(bottom=0.2)  # Adjust bottom space for readability

                        # Render the plot in Streamlit
                        st.pyplot(plt)
                        
                    elif selected_visualization_method == "Table":
                        # Show current missing values
                        st.dataframe(missing_values_df)
                    else:
                        st.error("Invalid visualization method selected!")

            columns_with_missing = missing_values_df[missing_values_df['Number Of Missing Entries'] > 0]['Column Name'].tolist()

            # Removed Machine Learning Imputation     
                             
            # Basic Imputation
            with st.expander("### Basic Imputation"):
                # Helper function to ensure a column is numeric, catch exceptions for non-numeric data        
                def ensure_numeric(df, column):
                    try:
                        if pd.api.types.is_numeric_dtype(df[column]):
                            return df[column]
                    except Exception as e:
                        st.error(f"The Column '{column}' has an incompatible data type for the selected technique to handle missing values: {e}")
                    return df[column]  # Return even if failed, so handling continues

                # Helper function to apply imputation to each missing value in a column
                def impute_values(df, column, method):
                    try:
                        if method == "Mean":
                            df[column] = ensure_numeric(df, column)
                            df[column].fillna(df[column].mean(), inplace=True)
                        elif method == "Median":
                            df[column] = ensure_numeric(df, column)
                            df[column].fillna(df[column].median(), inplace=True)
                        elif method == "Mode":
                            df[column].fillna(df[column].mode()[0], inplace=True)
                        elif method == "None":
                            df[column].fillna('None', inplace=True)
                        
                        # Extend handling for placeholders
                        df[column].replace(['', 'none', 'missing', 'na', 'not applicable', 'null', -9999, -999, 999, 9999, np.inf, -np.inf, np.nan], np.nan, inplace=True)
                    except Exception as e:
                        st.error(f"Error applying {method} imputation on column '{column}': {e}")
                    return df
            
        
    
                selected_column = st.selectbox("Select A Column To Handle Missing Values:", columns_with_missing)
                method = st.selectbox("Fill Missing Values With:", ("None", "Mean", "Median", "Mode"))

                if st.button("Apply Basic Imputation"):
                    # Save current state for undo
                    st.session_state.history.append(st.session_state.df.copy(deep=True))

                    # Apply imputation
                    st.session_state.df = impute_values(st.session_state.df, selected_column, method)

                    st.success(f"Filled missing values in '{selected_column}' using '{method}'.")

                    # Recalculate and show updated missing values
                    missing_values_df = update_missing_values_df()
                    st.dataframe(missing_values_df)

                # Undo functionality
                if st.button("Undo Basic Imputation"):
                    if st.session_state.history:
                        st.session_state.df = st.session_state.history.pop()
                        st.success(f"Undo Basic Imputation for column '{selected_column}'.")
                    else:
                        st.warning("No more steps to undo!")

            # Forward/Backward Fill
            with st.expander("### Forward Fill and Backward Fill"):
                selected_column = st.selectbox("Select A Column To Handle Missing Values (Forward/Backward Fill):", columns_with_missing)
                method = st.selectbox("Fill Missing Values With:", ("Forward Fill", "Backward Fill"))

                if st.button("Apply Forward/Backward Fill"):
                    st.session_state.history.append(st.session_state.df.copy(deep=True))

                    # Ensure the missing values are consistently detected
                    st.session_state.df[selected_column].replace(
                        ['', 'none', 'missing', 'na', 'not applicable', 'null', -9999, -999, 999, 9999, np.inf, -np.inf, np.nan],
                        np.nan, inplace=True
                    )
                    try:
                        if method == "Forward Fill":
                            st.session_state.df[selected_column].fillna(method='ffill', inplace=True)
                        elif method == "Backward Fill":
                            st.session_state.df[selected_column].fillna(method='bfill', inplace=True)

                        st.success(f"Applied {method} to '{selected_column}'.")
                    except Exception as e:
                        st.error(f"Error applying {method}: {e}")

                    missing_values_df = update_missing_values_df()
                    st.dataframe(missing_values_df)

                # Undo functionality
                if st.button("Undo Forward/Backward Fill"):
                    if st.session_state.history:
                        st.session_state.df = st.session_state.history.pop()
                        st.success(f"Undo {method} Fill for column '{selected_column}'.")
                    else:
                        st.warning("No more steps to undo!")


            # Remove Rows or Columns with Missing Values
            with st.expander("### Remove Rows with Missing Values"):
                removal_method = st.selectbox("Remove Rows:", ("Remove Rows with Missing Values",))

                if st.button("Apply Removal"):
                    st.session_state.history.append(st.session_state.df.copy(deep=True))

                    if removal_method == "Remove Rows with Missing Values":
                        try:
                            # Replace missing values placeholders before removal
                            st.session_state.df.replace(
                                ['', 'none', 'missing', 'na', 'not applicable', 'null', -9999, -999, 999, 9999, np.inf, -np.inf, np.nan],
                                np.nan, inplace=True
                            )
                            st.session_state.df.dropna(how='any', inplace=True)
                            st.success(f"Applied {removal_method}.")

                        except Exception as e:
                            st.error(f"Error applying {removal_method}: {e}")

                    missing_values_df = update_missing_values_df()
                    st.dataframe(missing_values_df)

                # Undo functionality
                if st.button("Undo Removal"):
                    if st.session_state.history:
                        st.session_state.df = st.session_state.history.pop()
                        st.success(f"Undo Removal.")
                    else:
                        st.warning("No more steps to undo!")
                        
            with st.expander("### Data Preview After Handling Missing Values"):
                st.dataframe(st.session_state.df)
                        
    handleMissingValues()
    
    def handleDuplicates():        
        # 3. Handling Duplicates
        st.write("### Handling Duplicates")
        if st.checkbox("Duplicates"):
            # a) Identifying Duplicates
            st.write("##### Identifying Duplicates")
            
            # i) Exact Matches
            with st.expander("Identifying Exact Duplicates"):
                exact_duplicates = st.session_state.df[st.session_state.df.duplicated(keep=False)]
                st.write(f"**Exact Duplicates**: {exact_duplicates.shape[0]} entries found")
                st.write("Showing Exact Duplicates")
                st.dataframe(exact_duplicates)


            # ii) Fuzzy Matching
            with st.expander("Identifying Almost Similar Duplicates"):
                # Split columns into string and numerical data types
                string_columns = st.session_state.df.select_dtypes(include=["object", "string"]).columns
                numerical_columns = st.session_state.df.select_dtypes(include=["number"]).columns

                # Ensure there are columns available for fuzzy matching
                if len(string_columns) == 0 and len(numerical_columns) == 0:
                    st.warning("No columns available for fuzzy matching.")
                else:
                    # Choose the column for fuzzy matching based on its data type
                    match_type = st.radio("Choose the type of column for fuzzy matching", ["String", "Numerical"])

                    if match_type == "String" and len(string_columns) > 0:
                        # String Fuzzy Matching
                        column_to_match = st.selectbox("Select column for fuzzy matching (String)", string_columns)

                        # Store the original data types of the entire DataFrame
                        original_dtypes = st.session_state.df.dtypes

                        # Convert selected column to string for fuzzy matching
                        st.session_state.df[column_to_match] = st.session_state.df[column_to_match].astype(str)

                        # Slider to set the similarity threshold for strings
                        threshold = st.slider("Set similarity threshold for fuzzy matching (String)", 70, 100, 90)

                        # Perform fuzzy matching
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

                        st.write(f"**Fuzzy Matches (String)**: {fuzzy_dup_df.shape[0]} pairs found")
                        st.dataframe(fuzzy_dup_df)

                        # Restore the original data types after fuzzy matching is done
                        st.session_state.df = st.session_state.df.astype(original_dtypes)

                        #st.write("Original data types have been restored.")
                        #st.dataframe(pd.DataFrame(st.session_state.df.dtypes, columns=["Data Type"]))

                    elif match_type == "Numerical" and len(numerical_columns) > 0:
                        # Numerical Fuzzy Matching
                        column_to_match = st.selectbox("Select column for fuzzy matching (Numerical)", numerical_columns)

                        # Slider to set the tolerance for numerical matching
                        tolerance = st.slider("Set tolerance for numerical matching (in percentage)", 0.0, 10.0, 5.0)

                        # Perform tolerance-based fuzzy matching for numerical data
                        matches = []
                        numerical_column = st.session_state.df[column_to_match]
                        for i, val in enumerate(numerical_column):
                            for j in range(i + 1, len(numerical_column)):
                                val_2 = numerical_column.iloc[j]
                                if pd.notnull(val) and pd.notnull(val_2):
                                    # Check if the absolute difference is within the tolerance percentage
                                    if np.abs(val - val_2) / val <= tolerance / 100:
                                        matches.append((i, j, val, val_2, np.abs(val - val_2)))

                        # Create a DataFrame with the matches
                        numerical_fuzzy_df = pd.DataFrame(matches, columns=["Row 1", "Row 2", "Value 1", "Value 2", "Difference"])

                        st.write(f"**Fuzzy Matches (Numerical)**: {numerical_fuzzy_df.shape[0]} pairs found")
                        st.dataframe(numerical_fuzzy_df)

                    else:
                        st.warning("No columns available for the selected data type.")

            # b) Handling Duplicates
            st.write("##### Aggregating Duplicates")
            # i) Aggregating Data
            with st.expander("Aggregating Exact Duplicates"):
                # Save the current state before applying the aggregation for undo functionality
                st.session_state.history.append(st.session_state.df.copy())

                # Filter columns by data types
                numerical_columns = st.session_state.df.select_dtypes(include=["number"]).columns
                string_columns = st.session_state.df.select_dtypes(include=["object"]).columns
                boolean_columns = st.session_state.df.select_dtypes(include=["bool"]).columns
                datetime_columns = st.session_state.df.select_dtypes(include=["datetime"]).columns

                # Perform aggregation only on duplicates in the selected column
                def perform_aggregation(agg_column, agg_method):
                    df = st.session_state.df.copy()
                    
                    # Find duplicate entries in the chosen column
                    duplicates_mask = df[agg_column].duplicated(keep=False)  # Mask to identify all duplicates
                    
                    # Apply aggregation to only the duplicate rows in the chosen column
                    if agg_column in numerical_columns:
                        if agg_method == "Sum":
                            df.loc[duplicates_mask, agg_column] += df[agg_column].rank(method="first")[duplicates_mask]
                        elif agg_method == "Mean":
                            mean_value = df.loc[duplicates_mask, agg_column].mean()
                            df.loc[duplicates_mask, agg_column] = mean_value
                        elif agg_method == "Count":
                            count_value = df.loc[duplicates_mask, agg_column].count()
                            df.loc[duplicates_mask, agg_column] = count_value
                        elif agg_method == "First":
                            df.loc[duplicates_mask, agg_column] = df.loc[duplicates_mask, agg_column].iloc[0]
                        elif agg_method == "Last":
                            df.loc[duplicates_mask, agg_column] = df.loc[duplicates_mask, agg_column].iloc[-1]
                        else:
                            st.warning("Invalid aggregation method.")
                            
                    elif agg_column in string_columns:
                        if agg_method == "Concatenate":
                            df.loc[duplicates_mask, agg_column] = df.loc[duplicates_mask, agg_column].apply(lambda x: f"{x}_{duplicates_mask.sum()}")
                        elif agg_method == "First":
                            df.loc[duplicates_mask, agg_column] = df.loc[duplicates_mask, agg_column].iloc[0]
                        elif agg_method == "Last":
                            df.loc[duplicates_mask, agg_column] = df.loc[duplicates_mask, agg_column].iloc[-1]
                        else:
                            st.warning("Invalid aggregation method.")
                            
                    elif agg_column in boolean_columns:
                        if agg_method == "Any":
                            df.loc[duplicates_mask, agg_column] = df.loc[duplicates_mask, agg_column].any()
                        elif agg_method == "All":
                            df.loc[duplicates_mask, agg_column] = df.loc[duplicates_mask, agg_column].all()
                        elif agg_method == "Count":
                            count_value = df.loc[duplicates_mask, agg_column].count()
                            df.loc[duplicates_mask, agg_column] = count_value
                        else:
                            st.warning("Invalid aggregation method.")
                            
                    elif agg_column in datetime_columns:
                        if agg_method == "Min":
                            df.loc[duplicates_mask, agg_column] = df.loc[duplicates_mask, agg_column].min()
                        elif agg_method == "Max":
                            df.loc[duplicates_mask, agg_column] = df.loc[duplicates_mask, agg_column].max()
                        elif agg_method == "First":
                            df.loc[duplicates_mask, agg_column] = df.loc[duplicates_mask, agg_column].iloc[0]
                        elif agg_method == "Last":
                            df.loc[duplicates_mask, agg_column] = df.loc[duplicates_mask, agg_column].iloc[-1]
                        else:
                            st.warning("Invalid aggregation method.")

                    else:
                        st.write("The selected column has an incompatible data type for aggregation.")

                    # Update only the chosen column
                    st.session_state.df[agg_column] = df[agg_column]
                    st.success(f"Data aggregated using '{agg_method}' method on column '{agg_column}'.")

                    # Show the modified DataFrame
                    st.write("### Data Preview After Aggregating Duplicate Values")
                    st.dataframe(st.session_state.df)

                # Combine all columns for selection
                all_columns = numerical_columns.tolist() + string_columns.tolist() + boolean_columns.tolist() + datetime_columns.tolist()

                if not all_columns:
                    st.warning("No columns available for aggregation.")
                else:
                    agg_column = st.selectbox("Select column to Aggregate Duplicate Data", all_columns)

                    # Dynamically show aggregation methods based on column type
                    if agg_column in numerical_columns:
                        agg_method = st.selectbox("Choose Aggregation Method", ["Sum", "Mean", "Count", "First", "Last"])
                    elif agg_column in string_columns:
                        agg_method = st.selectbox("Choose Aggregation Method", ["Concatenate", "First", "Last"])
                    elif agg_column in boolean_columns:
                        agg_method = st.selectbox("Choose Aggregation Method", ["Any", "All", "Count"])
                    elif agg_column in datetime_columns:
                        agg_method = st.selectbox("Choose Aggregation Method", ["Min", "Max", "First", "Last"])
                    else:
                        st.write("No aggregation method available for the selected column.")

                    # Button to Aggregate Data    
                    if st.button("Apply Aggregation"):
                        perform_aggregation(agg_column, agg_method)

                # Undo button functionality
                if st.button("Undo Data Aggregation"):
                    if st.session_state.history:
                        st.session_state.df = st.session_state.history.pop()
                        st.success(f"Undo successful aggregation for column '{agg_column}'.")
                    else:
                        st.warning("No more steps to undo!")


                        
                    
            # ii) Removing Duplicates
            st.write("##### Removing Duplicates")
            with st.expander("Removing Exact Duplicates"):
                # Save the current state before applying the removal for undo functionality
                st.session_state.history.append(st.session_state.df.copy())
                
                st.session_state.df.drop_duplicates(inplace=True)
                st.success("Exact duplicates removed.")
                st.write("Data Preview After Removing Duplicates")
                st.dataframe(st.session_state.df)
                    
            # Undo button functionality
            if st.button("Undo Duplicate Removal"):
                if st.session_state.history:
                    st.session_state.df = st.session_state.history.pop()
                    st.success("Undo Duplicate Removal.")
                else:
                    st.warning("No more steps to undo!")
                    

        
    handleDuplicates()        

    def outlierDetection():
        st.write("### Handling Outliers")
        numerical_columns = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        outlier_summary = {}
        # Outlier Detection Summary Table
        if st.checkbox("Outliers Summary"):
            with st.expander("Outliers Detection Per Column"):
                # Detecting outliers for each numerical column
                for col in numerical_columns:
                    Q1 = st.session_state.df[col].quantile(0.25)
                    Q3 = st.session_state.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = st.session_state.df[(st.session_state.df[col] < (Q1 - 1.5 * IQR)) | (st.session_state.df[col] > (Q3 + 1.5 * IQR))]
                    outlier_summary[col] = len(outliers)

                outlier_summary_df = pd.DataFrame(list(outlier_summary.items()), columns=['Column', 'Number of Outliers'])
                st.write("Outlier Summary")
                st.dataframe(outlier_summary_df)

            # Visualization Options
            with st.expander("Outlier Visualization"):
                vis_option = st.selectbox("Choose Visualization Type", ["Select", "Box Plot", "Scatter Plot"])

                try:
                    if vis_option == "Box Plot":
                        numerical_col = st.selectbox("Select Numerical Column", numerical_columns, key="boxplot_num_col")
                        categorical_col = st.selectbox("Select Categorical Column (Optional)", st.session_state.df.select_dtypes(include="category").columns.tolist(), key="boxplot_cat_col", index=0)

                        if st.button("Show Box Plot"):
                            plt.figure(figsize=(10, 6))
                            if categorical_col:
                                sns.boxplot(x=categorical_col, y=numerical_col, data=st.session_state.df)
                            else:
                                sns.boxplot(y=numerical_col, data=st.session_state.df)
                            st.pyplot(plt)

                    elif vis_option == "Scatter Plot":
                        x_axis_col = st.selectbox("Select X-Axis Numerical Column", numerical_columns, key="scatter_x")
                        y_axis_col = st.selectbox("Select Y-Axis Numerical Column", numerical_columns, key="scatter_y")
                        categorical_col = st.selectbox("Select Categorical Column (Optional)", st.session_state.df.select_dtypes(include="category").columns.tolist(), key="scatter_cat", index=0)

                        if st.button("Show Scatter Plot"):
                            plt.figure(figsize=(10, 6))
                            if categorical_col:
                                sns.scatterplot(x=x_axis_col, y=y_axis_col, hue=categorical_col, data=st.session_state.df)
                            else:
                                sns.scatterplot(x=x_axis_col, y=y_axis_col, data=st.session_state.df)
                            st.pyplot(plt)

                except Exception as e:
                    st.error(f"Error occurred: {e}")
                    st.error("Ensure that your selected columns have the correct data types and are suitable for the chosen visualization.")

            # Outlier Handling Options
            with st.expander("Outlier Handling"):
                st.write("Select Outlier Handling Method")

                col = st.selectbox("Select Column to Handle Outliers", numerical_columns, key="outlier_col")
                method = st.selectbox("Choose Method", ["Remove", "Cap", "Transform"])

                if st.button("Apply Outlier Handling Method"):
                    try:
                        # Save the current state for undo functionality
                        st.session_state.history.append(st.session_state.df.copy())

                        Q1 = st.session_state.df[col].quantile(0.25)
                        Q3 = st.session_state.df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        if method == "Remove":
                            st.session_state.df = st.session_state.df[~((st.session_state.df[col] < lower_bound) | (st.session_state.df[col] > upper_bound))]
                            st.success(f"Outliers removed based on IQR for column '{col}'.")
                        
                        elif method == "Cap":
                            st.session_state.df[col] = np.where(st.session_state.df[col] < lower_bound, lower_bound, st.session_state.df[col])
                            st.session_state.df[col] = np.where(st.session_state.df[col] > upper_bound, upper_bound, st.session_state.df[col])
                            st.success(f"Outliers capped for column '{col}'.")

                        elif method == "Transform":
                            transformation_type = st.selectbox("Choose Transformation", ["Log", "Square Root", "Box-Cox"], key="transform_type")

                            if transformation_type == "Log":
                                st.session_state.df[col] = np.log1p(st.session_state.df[col].clip(lower=1e-9))
                                st.success(f"Log Transformation applied to column '{col}'.")

                            elif transformation_type == "Square Root":
                                st.session_state.df[col] = np.sqrt(st.session_state.df[col].clip(lower=0))
                                st.success(f"Square Root Transformation applied to column '{col}'.")

                            elif transformation_type == "Box-Cox":
                                st.session_state.df[col], _ = boxcox(st.session_state.df[col].clip(lower=1e-9))
                                st.success(f"Box-Cox Transformation applied to column '{col}'.")

                        st.dataframe(st.session_state.df)

                    except Exception as e:
                        st.error(f"Error occurred during outlier handling: {e}")

                # Undo Button for Outlier Handling
                if st.button("Undo Outlier Handling"):
                    if st.session_state.history:
                        st.session_state.df = st.session_state.history.pop()
                        st.success("Undo successful for outlier handling.")
                    else:
                        st.warning("No more steps to undo.")

    outlierDetection()

                    
    def normalization():
        st.write("### Data Normalization and Scaling")
        numerical_columns = st.session_state.df.select_dtypes(include=np.number).columns
        summary = []
        
        # Display data distribution summary
        if st.checkbox("Normalization and Scaling"):
            with st.expander("Show Data Distribution Summary"):
                for col in numerical_columns:
                    data = st.session_state.df[col].dropna()
                    
                    # Calculating skewness and kurtosis
                    skew_value = skew(data)
                    kurtosis_value = kurtosis(data) + 3  # Pearson definition
                    
                    # Interpret skewness
                    if -0.5 <= skew_value <= 0.5:
                        skew_desc = "Fairly symmetric"
                    elif -1 <= skew_value < -0.5 or 0.5 < skew_value <= 1:
                        skew_desc = "Moderate skew"
                    else:
                        skew_desc = "Highly skewed"

                    # Interpret kurtosis
                    if kurtosis_value < 3:
                        kurt_desc = "Platykurtic (light tails, fewer extreme values)"
                    elif kurtosis_value == 3:
                        kurt_desc = "Mesokurtic (normal, moderate tails)"
                    else:
                        kurt_desc = "Leptokurtic (heavy tails, frequent extreme values)"

                    # Append summary with description fields
                    summary.append({
                        "Column": col,
                        "Min": data.min(),
                        "Max": data.max(),
                        "Skewness": skew_value,
                        "Skewness Description": skew_desc,
                        "Kurtosis": kurtosis_value,
                        "Kurtosis Description": kurt_desc,
                        "Outliers (IQR)": ((data < data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25))) | 
                                            (data > data.quantile(0.75) + 1.5 * (data.quantile(0.75) - data.quantile(0.25)))).sum()
                    })
                    
                summary_df = pd.DataFrame(summary)
                st.write("Summary of Data Distribution, Range, and Outliers:")
                st.dataframe(summary_df)
            
            # Normalization Options
            with st.expander("Normalization And Visualization Options"):
                st.write("##### Choose Normalization Method and Columns")
                method = st.selectbox("Choose Normalization Method", ["None", "Z-score", "MinMax", "Robust"])
                columns = st.multiselect("Select Columns to Normalize/Scale", numerical_columns)
                
                # Visualization Options
                st.write("##### Data Visualization")
                visualization_method = st.selectbox("Choose Visualization Method", ["Histogram", "Box Plot", "Density Plot (KDE)"])
                y_col = st.selectbox("Select Y Column (Numerical)", numerical_columns)
                
                # Additional visualization parameters
                def calculate_bin_parameters(column_data):
                    n = len(column_data)
                    iqr = np.percentile(column_data, 75) - np.percentile(column_data, 25)
                    std_dev = np.std(column_data)
                    min_value, max_value = column_data.min(), column_data.max()
                    
                    bin_widths = {
                        'Square Root Rule': np.ceil(np.sqrt(n)),
                        'Sturges’ Rule': np.ceil(1 + 3.322 * np.log10(n)),
                        'Freedman-Diaconis Rule': np.ceil(2 * (iqr / (n ** (1/3)))),
                        'Scott’s Rule': np.ceil(3.5 * (std_dev / (n ** (1/3))))
                    }

                    bin_params = []
                    for rule, width in bin_widths.items():
                        num_bins = int(np.ceil((max_value - min_value) / width))
                        bin_params.append({
                            'Rule': rule,
                            'Bin Width': width,
                            'Number of Bins': num_bins,
                            'Range Min': min_value - 1,
                            'Range Max': max_value + 1
                        })
                        
                    return bin_params

                if visualization_method == "Histogram":
                    bin_params = calculate_bin_parameters(st.session_state.df[y_col])
                    selected_rule = st.selectbox("Select Bin Width Calculation Rule", [param['Rule'] for param in bin_params])
                    selected_param = next((param for param in bin_params if param['Rule'] == selected_rule), bin_params[0])
                    bin_width = st.slider("Bin Width", min_value=1, max_value=int(selected_param['Bin Width']), step=1, value=int(selected_param['Bin Width']))
                    num_bins = st.slider("Number of Bins", min_value=1, max_value=100, step=1, value=int(selected_param['Number of Bins']))
                    range_min = st.slider("Range Minimum", float(selected_param['Range Min']), float(selected_param['Range Max']), value=float(selected_param['Range Min']))
                    range_max = st.slider("Range Maximum", float(selected_param['Range Min']), float(selected_param['Range Max']), value=float(selected_param['Range Max']))

                elif visualization_method == "Box Plot":
                    categorical_col = st.selectbox("Select Categorical Column (Optional)", st.session_state.df.select_dtypes(include="category").columns.tolist())

                # Normalize and visualize data
                if st.button("Apply Normalization and Plot"):
                    try:
                        st.session_state.history.append(st.session_state.df.copy())

                        # Apply normalization
                        if method == "Z-score":
                            st.session_state.df[columns] = StandardScaler().fit_transform(st.session_state.df[columns])
                        elif method == "MinMax":
                            st.session_state.df[columns] = MinMaxScaler().fit_transform(st.session_state.df[columns])
                        elif method == "Robust":
                            st.session_state.df[columns] = RobustScaler().fit_transform(st.session_state.df[columns])

                        # Plot original and normalized data side-by-side
                        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                        fig.suptitle("Data Distribution: Before and After Normalization")

                        if visualization_method == "Histogram":
                            sns.histplot(st.session_state.history[-1][y_col], bins=num_bins, kde=False, ax=axes[0], binwidth=bin_width, binrange=(range_min, range_max))
                            axes[0].set_title("Before Normalization")
                            sns.histplot(st.session_state.df[y_col], bins=num_bins, kde=False, ax=axes[1], binwidth=bin_width, binrange=(range_min, range_max))
                            axes[1].set_title("After Normalization")

                        elif visualization_method == "Box Plot":
                            sns.boxplot(x=categorical_col, y=y_col, data=st.session_state.history[-1], ax=axes[0])
                            axes[0].set_title("Before Normalization")
                            sns.boxplot(x=categorical_col, y=y_col, data=st.session_state.df, ax=axes[1])
                            axes[1].set_title("After Normalization")

                        elif visualization_method == "Density Plot (KDE)":
                            sns.kdeplot(st.session_state.history[-1][y_col], ax=axes[0])
                            axes[0].set_title("Before Normalization")
                            sns.kdeplot(st.session_state.df[y_col], ax=axes[1])
                            axes[1].set_title("After Normalization")

                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error occurred: {e}")
                        
                # Undo Button
                if 'history' not in st.session_state:
                    st.session_state.history = []

                if st.button("Undo Last Normalization"):
                    if st.session_state.history:
                        st.session_state.df = st.session_state.history.pop()
                        st.success("Successfully reverted the last normalization step.")
                    else:
                        st.warning("No previous normalization steps to undo.")

            # Add Pair Plot for All Numerical Columns
            with st.expander("View Pair Plot for All Numerical Columns"):
                st.write("##### Pair Plot of Selected Numerical Columns")
                pairplot_color = st.selectbox("Select Categorical Column for Color (Optional)", [None] + st.session_state.df.select_dtypes(include="category").columns.tolist())
                selected_pairplot_columns = st.multiselect("Select Numerical Columns for Pair Plot", numerical_columns)
                
                if st.button("Show Pair Plot"):
                    try:
                        if selected_pairplot_columns:
                            # Prepare pair plot based on selected columns and optional color
                            if pairplot_color:
                                pair_plot_fig = sns.pairplot(st.session_state.df[selected_pairplot_columns + [pairplot_color]], hue=pairplot_color, diag_kind='kde')
                            else:
                                pair_plot_fig = sns.pairplot(st.session_state.df[selected_pairplot_columns], diag_kind='kde')
                            
                            st.pyplot(pair_plot_fig)
                        else:
                            st.warning("Please select at least one numerical column to generate the pair plot.")

                    except Exception as e:
                        st.error(f"Error occurred in generating pair plot: {e}")
    normalization()



    def textConsistency():
        # 6a. Text Data Consistency
        st.write("### Handling Text Data Consistency")
        if st.checkbox("Text Data Consistency"):
            with st.expander("Text Consistency"):     
                st.write("Automatically Converting All Text Columns To Lowercase Unless Altered Here.")
                # Provide option to change case
                col = st.selectbox("Select Text Column To Modify:", st.session_state.df.select_dtypes(include='string').columns)
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
                        
    textConsistency()    

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
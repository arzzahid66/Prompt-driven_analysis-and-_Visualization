import pandas as pd
import streamlit as st
import os 
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from dotenv import load_dotenv
from pandasai import PandasAI
from pandasai.llm import GooglePalm
from llama_index.query_engine import PandasQueryEngine
from IPython.display import Markdown , displays
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.graph_objects as go
import plotly.express as px
import sys
import io
load_dotenv()

API_KEY =st.sidebar.text_input("Enter your OpenAI API key", type="password")
if st.sidebar.button("enter"):
    llm = OpenAI(api_token=API_KEY)
    pandas_ai = PandasAI(llm)
else:
    st.write("sorry enter api key")



# Function to save sidebar outputs to session state
def save_sidebar_outputs():
    if 'df_shape' not in st.session_state:
        st.session_state.df_shape = None
    if 'df_dtypes' not in st.session_state:
        st.session_state.df_dtypes = None
    if 'df_columns' not in st.session_state:
        st.session_state.df_columns = None
    if 'df_describe' not in st.session_state:
        st.session_state.df_describe = None
    if 'cols_to_drop' not in st.session_state:
        st.session_state.cols_to_drop = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None

st.title("Prompt-driven analysis and Visualization by Abdul_Rehman_Zahid")

uploaded_file = st.file_uploader("Upload your data file for analysis", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension in ['xlsx', 'xls']:
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        st.stop()

    # Check if there are columns named "Unnamed"
    if any(col.startswith('Unnamed') for col in df.columns):
        # Rename columns with "Unnamed" prefix
        df.columns = [f"Column_{i}" if col.startswith("Unnamed") else col for i, col in enumerate(df.columns)]

    if 'original_df' not in st.session_state or st.session_state.original_df is None:
        st.session_state.original_df = df.copy()
    else:
        df = st.session_state.original_df.copy()

    st.write(df.head(3))
    prompt = st.text_area("Enter prompt", height=100)

    if st.button("Generate"):
        if prompt:
            with st.spinner("Response is generating"):
                df = SmartDataframe(df, config={"llm": llm})
                st.text(df.chat(prompt))
        else:
            st.warning("Please enter prompt")

    save_sidebar_outputs()

    about_data_button = st.sidebar.button("Basic Info About Data")
    if about_data_button:
        st.session_state.df_shape = df.shape
        st.session_state.df_dtypes = df.dtypes
        st.session_state.df_columns = df.columns
        st.session_state.df_isnull = df.isnull().sum()
        st.session_state.df_unique = df.nunique()

    if 'df_shape' in st.session_state and st.session_state.df_shape is not None:
        st.sidebar.subheader("Shape of the Data:")
        st.sidebar.text(st.session_state.df_shape)
    if 'df_dtypes' in st.session_state and st.session_state.df_dtypes is not None:
        st.sidebar.subheader("Data Types:")
        st.sidebar.write(st.session_state.df_dtypes)
    if 'df_columns' in st.session_state and st.session_state.df_columns is not None:
        st.sidebar.subheader("Columns in the Dataset:")
        st.sidebar.write(st.session_state.df_columns)
    if 'df_isnull' in st.session_state and st.session_state.df_isnull is not None:
        st.sidebar.subheader("Null vales in Data:")
        st.sidebar.write(st.session_state.df_isnull)
    if 'df_unique' in st.session_state and st.session_state.df_unique is not None:
        st.sidebar.subheader("Unique values in Data :")
        st.sidebar.write(st.session_state.df_unique)
    
    # Step 1: Remove selected columns
    if st.session_state.cols_to_drop is None:
        st.session_state.cols_to_drop = []
    st.title("Drop Columns")
    cols_to_drop = st.multiselect(" Select column to drop", options=df.columns, default=[col for col in st.session_state.cols_to_drop if col in df.columns])
    if cols_to_drop:
        if st.button("Remove columns"):
            df = df.drop(columns=cols_to_drop)
            st.session_state.cols_to_drop = cols_to_drop
            st.session_state.original_df = df.copy()  # Update original DataFrame
            st.success(f"Columns {', '.join(cols_to_drop)} dropped successfully!")
        else:
            st.warning("Please select columns to drop")
    if st.button("see result"):
        st.write(df.columns)
    # Step 2: Remove null values from selected columns
    st.title("Drop Null values")
    col_to_drop_null = st.multiselect("Select column to drop null values", options=df.columns)
    if col_to_drop_null:
        if st.button("Remove null values"):
            for col in col_to_drop_null:
                if col in df.columns:
                    df = df.dropna(subset=[col])
                    st.success(f"Null values in '{col}' dropped successfully!")
            st.session_state.original_df = df.copy()  # Update original DataFrame
        if st.button("see_result"):
            st.write(df.isnull().sum())
    # Step 3: Plot bar chart for selected column
    st.title("Histogram plot")
    col_to_plot = st.selectbox("Step 3: Select column for histogram plot", options=df.columns)
    if col_to_plot:
        if st.button("Histogram Plot"):
            plt.figure(figsize=(4,2))
            df[col_to_plot].plot(kind='hist')
            plt.xlabel(col_to_plot)
            plt.ylabel('Frequency')
            plt.title(f'Histogram plot of {col_to_plot}')
            st.pyplot(plt)
            st.session_state.original_df = df.copy()  # Update original DataFrame
    # Step 4: Create boxplot for selected column
    st.title("Boxplot")
    col_to_plot_boxplot = st.selectbox("Select column for boxplot", options=df.columns)
    if col_to_plot_boxplot:
        if st.button("Boxplot"):
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=df[col_to_plot_boxplot])
            plt.xlabel(col_to_plot_boxplot)
            plt.title(f'Boxplot of {col_to_plot_boxplot}')
            st.pyplot(plt)
            st.session_state.original_df = df.copy()  # Update original DataFrame
        
    # Step 5: Calculate Interquartile Range (IQR) and print it
    if st.button("Calculate and Print IQR"):
       Q1 = df.quantile(0.25)
       Q3 = df.quantile(0.75)
       IQR = Q3 - Q1
       st.write("Interquartile Range (IQR):")
       st.write(IQR)
    # Step 6: Calculate correlation matrix
    if st.button("Calculate Correlation Matrix"):
       correlation_matrix = df.corr()
       st.write("Correlation Matrix:")
       st.write(correlation_matrix)
    # Step 7: Display bar plot of two variables
    st.title("Create Scatter plot")
    selected_columns = st.multiselect("Select two columns for scatter plot first select Y axis", options=df.columns)
    hue_column = st.selectbox("Select a column for hue (optional)", options=df.columns, index=0)
    if len(selected_columns) == 2:
        if st.button("Display scatter Plot"):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=selected_columns[1], y=selected_columns[0],hue=hue_column, data=df)
            plt.title(f'Line scatter of {selected_columns[0]} vs {selected_columns[1]}')
            plt.legend(title=hue_column)
            st.pyplot(plt)
            st.session_state.original_df = df.copy() 
    # Step 8: Display bar plot of two variables
    st.title("Create Bar plot")
    selected_column_barplot = st.multiselect("Get top 5 values in bar plot  ", options=df.columns)
    st.write("Select Y value First :")
    if len(selected_column_barplot) == 2:
        if st.button("Display Bar Plot"):
            plt.figure(figsize=(10, 5))
            df[selected_column_barplot].value_counts().nlargest(5).plot(kind='bar')
            plt.xlabel(selected_column_barplot[1])
            plt.ylabel(selected_column_barplot[0])
            plt.title(f'Bar Plot of {selected_column_barplot}')
            st.pyplot(plt)
            st.session_state.original_df = df.copy()
    # Step 9: Display bar plot of two variables
    st.title("Create line plot")
    selected_columns = st.multiselect("Select two columns for line plot select First Y axis value ", options=df.columns)
    hue_options = [None] + list(df.columns)
    hue_column = st.selectbox("Select a column for hue", options=hue_options)
    if len(selected_columns) == 2:
        if st.button("Display lineplot Plot"):
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=selected_columns[1], y=selected_columns[0], hue=hue_column , style=hue_column, data=df)
            plt.title(f'Line scatter of {selected_columns[0]} vs {selected_columns[1]}')
            plt.legend(title=hue_column)
            st.pyplot(plt)
        
    
    # Select columns for the sunburst chart
    st.title("Create Sunbrust plot")
    selected_path_columns = st.multiselect("Select path columns", options=df.columns)
    selected_value_column = st.selectbox("Select values column", options=df.columns)
    selected_color_column = st.selectbox("Select color column", options=df.columns)

    # Check if at least one path column and the value column are selected
    if selected_path_columns and selected_value_column:
        button_clicked = st.button("Create Sunburst Chart")
        if button_clicked:
            # Create the Sunburst chart
            fig = px.sunburst(df, path=selected_path_columns, values=selected_value_column, color=selected_color_column)
            st.plotly_chart(fig)
             

    # Custom CSS to position the chatbot iframe on the right side and make it smaller
    st.markdown(
    """
    <style>
    .chatbot-iframe {
        width: 300px;
        height: 400px;
        position: fixed;
        bottom: 20px;
        right: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

# Embedding the chatbot iframe
    st.markdown(
    """
    <iframe class="chatbot-iframe" src="https://copilotstudio.microsoft.com/environments/Default-3077906c-1911-42d2-969c-4bc4a2832b7a/bots/Default_arzAssistant/webchat?__version__=2" frameborder="0"></iframe>
    """,
    unsafe_allow_html=True
    )
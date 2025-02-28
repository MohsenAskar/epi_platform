# 22_data_management.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

########################
# Page Title
########################
st.title("Data Management & Wrangling for Epidemiology")

st.write("""
*Welcome to an interactive module on data cleaning and wrangling. 
Whether you love data or find it tedious, let's explore some **friendly** 
and **visual** ways to manage real-world epidemiological data.*
""")

###############################################################################
# Sidebar / Data Upload
###############################################################################
st.sidebar.header("1. Upload or Select Data")
data_source = st.sidebar.selectbox(
    "Choose data source:",
    ["Use a sample dataset (COVID)", "Upload my own CSV"]
)

def load_sample_data():
    # Let's simulate a small 'COVID' dataset with typical epidemiology columns
    rng = np.random.default_rng(seed=42)
    n = 200
    data = {
        "ID": np.arange(1, n+1),
        "Age": rng.integers(18, 90, n),
        "Sex": rng.choice(["M", "F"], n),
        "Symptom_Onset": pd.date_range("2020-01-01", periods=n, freq='1D'),
        "Test_Result": rng.choice(["Positive", "Negative", None], n, p=[0.5, 0.4, 0.1]),
        "Contact_Tracing": rng.choice(["Yes", "No"], n, p=[0.7, 0.3]),
        "Severity": rng.choice(["Mild", "Moderate", "Severe", None], n, p=[0.4, 0.4, 0.15, 0.05])
    }
    df = pd.DataFrame(data)
    # Introduce some random missingness
    for col in ["Age", "Symptom_Onset"]:
        mask = rng.choice([True, False], size=n, p=[0.05, 0.95])
        df.loc[mask, col] = np.nan
    return df

if data_source == "Use a sample dataset (COVID)":
    df_main = load_sample_data()
    st.sidebar.success("Using sample COVID-like data.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df_main = pd.read_csv(uploaded_file)
        st.sidebar.success("Custom dataset uploaded!")
    else:
        st.sidebar.warning("Please upload a CSV or switch to the sample data.")
        df_main = None


###############################################################################
# Main Selectbox: Steps in Data Wrangling
###############################################################################
if df_main is not None:
    step = st.selectbox(
        "Select a Data Wrangling Step:",
        [
            "Data Preview & Basic Summaries",
            "Missing Data Exploration",
            "Recoding / Creating New Variables",
            "Reshaping (Wide ↔ Long)",
            "Merging Datasets",
            "Check / Remove Duplicates",
            "Renaming & Type Conversion",
            "Outlier Detection",
            "Data Validation"
        ]
    )

    ###########################################################################
    # Step 1: Data Preview & Basic Summaries
    ###########################################################################
    if step == "Data Preview & Basic Summaries":
        st.header("Data Preview & Basic Summaries")

        st.write("""
        **Step 1**: Always start by *getting to know* your dataset.
        - Check the first few rows
        - Examine data types
        - Generate descriptive statistics
        """)

        # Show top rows
        st.subheader("A Peek at the Data")
        st.dataframe(df_main.head(10))

        # Interactive dimension / column info
        st.write(f"**Data Dimensions**: {df_main.shape[0]} rows × {df_main.shape[1]} columns")

        with st.expander("Column Data Types"):
            st.write(df_main.dtypes.to_frame("Type"))

        with st.expander("Basic Descriptive Statistics"):
            st.write(df_main.describe(include='all'))

        st.info("""
        *Tip:* Data summaries can reveal suspicious values, 
        out-of-range numbers, or unexpected data types (like "Age" stored as text).
        """)

    ###########################################################################
    # Step 2: Missing Data Exploration
    ###########################################################################
    elif step == "Missing Data Exploration":
        st.header("Missing Data Exploration")

        st.write("""
        **Step 2**: Missing data is common in epidemiological studies. 
        Let's visualize and count missingness so we can decide on *imputation*, 
        *exclusion*, or *further queries*.
        """)

        # 1. Quick missingness table
        missing_counts = df_main.isna().sum()
        missing_percent = (missing_counts / len(df_main)) * 100
        missing_table = pd.DataFrame({
            "Missing Count": missing_counts,
            "Percent (%)": missing_percent.round(2)
        })

        st.subheader("Missingness by Column")
        st.dataframe(missing_table)

        # 2. Visual approach - e.g., color-coded heatmap of missing data
        #   We'll do a small sub-sample if dataset is large
        n_show = min(len(df_main), 200)  # to keep plot manageable
        df_sample = df_main.sample(n_show, random_state=42).reset_index(drop=True)

        # Convert booleans to int (1=missing, 0=not missing)
        miss_matrix = df_sample.isna().astype(int)

        st.subheader("Visual Missingness (sample of rows)")
        fig = px.imshow(
            miss_matrix.T,  # transpose so columns on y-axis
            color_continuous_scale=["#FFFFFF", "#FF0000"],
            aspect="auto",
            title="Red = Missing, White = Present (Sampled Rows)"
        )
        # We'll hide the axis tick labels for cleanliness
        fig.update_layout(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=True),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        *Tip:* If certain columns have *systematic* missingness (e.g., 
        "Severity" missing in mild cases only), that pattern might bias analyses.
        """)

    ###########################################################################
    # Step 3: Recoding / Creating New Variables
    ###########################################################################
    elif step == "Recoding / Creating New Variables":
        st.header("Recoding & Creating New Variables")

        st.write("""
        **Step 3**: We often need to recode categorical variables, 
        create new derived variables (like BMI from weight & height), 
        or categorize continuous variables (e.g., Age groups).
        """)

        st.subheader("Example: Categorizing 'Age'")
        df_main_copy = df_main.copy()

        # Let user pick cutoffs for age categories
        bins_str = st.text_input(
            "Enter age cutoffs (comma separated)",
            "0,18,40,65,100"
        )
        try:
            bins = [float(x.strip()) for x in bins_str.split(",")]
        except:
            bins = [0, 18, 40, 65, 100]

        labels = [f"{int(bins[i])}-{int(bins[i+1])-1}" for i in range(len(bins)-1)]

        if "Age" in df_main_copy.columns:
            df_main_copy["AgeGroup"] = pd.cut(df_main_copy["Age"], bins=bins, labels=labels, right=False)
            st.write("**Preview of New AgeGroup Column**")
            st.dataframe(df_main_copy[["Age","AgeGroup"]].head(10))
        else:
            st.error("No 'Age' column found. Try the sample dataset or rename your columns accordingly.")

        st.subheader("Visualizing the Recoded Variable")
        if "AgeGroup" in df_main_copy.columns:
            fig = px.histogram(
                df_main_copy, x="AgeGroup", color="Sex",
                title="Distribution of Age Groups by Sex",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info("""
        *Tip:* Keep a **data dictionary** listing each new variable, its meaning, 
        and possible values. This helps avoid confusion later.
        """)

    ###########################################################################
    # Step 4: Reshaping (Wide ↔ Long)
    ###########################################################################
    elif step == "Reshaping (Wide ↔ Long)":
        st.header("Reshaping Data (Wide ↔ Long)")

        st.write("""
        **Step 4**: Epidemiological data sometimes comes in 'wide' format 
        (one row per patient, multiple columns for repeated measurements) 
        or 'long' format (one row per measurement). We often need to pivot 
        from one to the other.
        """)

        st.subheader("Example: Pivoting Mild/Moderate/Severe into separate columns")

        if "Severity" not in df_main.columns:
            st.error("No 'Severity' column found to pivot. Use the sample dataset to see an example.")
        else:
            # Let's do a simple pivot: count how many people in each severity category per 'Sex'
            pivot_table = df_main.pivot_table(
                index="Sex", 
                columns="Severity", 
                values="ID", 
                aggfunc="count"
            ).fillna(0)

            st.write("**Pivot Table**: # of IDs by Sex × Severity")
            st.dataframe(pivot_table)

            st.write("""
            **Long Format**: Alternatively, each row can represent 
            (Sex, Severity, count). 
            """)
            pivot_long = pivot_table.reset_index().melt(
                id_vars="Sex", 
                var_name="Severity", 
                value_name="Count"
            )
            st.dataframe(pivot_long)

            fig = px.bar(
                pivot_long,
                x="Sex", y="Count", color="Severity",
                title="Counts by Sex and Severity (Long Format)"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info("""
        *Tip:* Reshaping is crucial if you have repeated measures over time 
        or multiple lab tests per individual. Decide if your analysis needs 
        "long" or "wide" data to run the correct procedures in your stats software.
        """)

    ###########################################################################
    # Step 5: Merging Datasets
    ###########################################################################
    elif step == "Merging Datasets":
        st.header("Merging Datasets")

        st.write("""
        **Step 5**: Often in epidemiology, we combine multiple files, 
        e.g., linking lab data with demographic data. Let's simulate a second 
        dataset and merge it with our main data on a common key.
        """)

        st.subheader("Creating a Second Dataset: 'Lab Results'")
        rng = np.random.default_rng(2023)
        # We'll choose a subset of IDs, simulating that not everyone has lab data
        if "ID" not in df_main.columns:
            st.error("No 'ID' column found. Sample dataset includes 'ID' for merges. Try that.")
        else:
            # We'll pick half the IDs to have lab data
            unique_ids = df_main["ID"].dropna().unique()
            subset_ids = rng.choice(unique_ids, size=len(unique_ids)//2, replace=False)

            # Simulate lab results
            data_lab = {
                "ID": subset_ids,
                "CRP": rng.normal(loc=10, scale=5, size=len(subset_ids)).round(2),  # C-Reactive Protein
                "WBC": rng.normal(loc=7, scale=2, size=len(subset_ids)).round(1)   # White Blood Cell count
            }
            df_lab = pd.DataFrame(data_lab)

            st.write("**Lab Dataset (first few rows)**")
            st.dataframe(df_lab.head(10))

            st.subheader("Merging on 'ID'")
            merge_type = st.selectbox(
                "Select merge type:",
                ["left", "right", "inner", "outer"],
                help="""
                - left: keep all rows in left (main) dataset, matching from right
                - right: keep all rows in right (lab) dataset
                - inner: keep only matching IDs
                - outer: keep all rows from both
                """
            )

            df_merged = pd.merge(
                df_main, df_lab,
                on="ID", how=merge_type
            )

            st.write(f"**Merged Dataset**: {len(df_merged)} rows")
            st.dataframe(df_merged.head(10))

            st.info("""
            *Tip:* Always check for duplicates, mismatched keys, and 
            row counts after merging. An unexpected row count might 
            indicate data entry errors or multiple merges per ID.
            """)
    ###########################################################################
    # Step 6: Check duplicates
    ###########################################################################
    elif step == "Check / Remove Duplicates":
        st.header("Check for Duplicates")
        duplicate_count = df_main.duplicated().sum()
        st.write(f"Number of completely duplicated rows: {duplicate_count}")

        if duplicate_count > 0:
            remove_dup = st.checkbox("Remove duplicates?", value=False)
            if remove_dup:
                df_main.drop_duplicates(inplace=True)
                st.success("Duplicates removed!")
    ###########################################################################
    # Step 7: Rename, conveert variable type
    ###########################################################################
    elif step == "Renaming & Type Conversion":
        col_to_rename = st.selectbox("Select column to rename", df_main.columns)
        new_name = st.text_input("New column name", "")
        if st.button("Rename"):
            df_main.rename(columns={col_to_rename: new_name}, inplace=True)
            st.write("Renamed successfully!")
            
        col_to_convert = st.selectbox("Select column to convert", df_main.columns)
        target_type = st.selectbox("Target type", ["int", "float", "str", "datetime"])
        if st.button("Convert Type"):
            try:
                if target_type == "int":
                    df_main[col_to_convert] = df_main[col_to_convert].astype(int)
                elif target_type == "float":
                    df_main[col_to_convert] = df_main[col_to_convert].astype(float)
                # etc.
                st.success(f"Converted {col_to_convert} to {target_type}")
            except Exception as e:
                st.error(f"Conversion failed: {e}")
    ###########################################################################
    # Step 8: Outlier detection
    ###########################################################################
    elif step == "Outlier Detection":
        numeric_cols = df_main.select_dtypes(include=[np.number]).columns.tolist()
        col_outlier = st.selectbox("Choose numeric column", numeric_cols)

        fig = px.box(df_main, y=col_outlier, points="all", title="Boxplot for Outliers")
        st.plotly_chart(fig, use_container_width=True)
        
        threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0)
        col_mean = df_main[col_outlier].mean()
        col_std = df_main[col_outlier].std()
        df_main["zscore"] = (df_main[col_outlier] - col_mean) / col_std
        outliers = df_main[abs(df_main["zscore"]) > threshold]
        st.write(f"Found {len(outliers)} outliers above threshold {threshold}.")

        remove_outliers = st.button("Remove Outliers")
        if remove_outliers:
            df_main.drop(outliers.index, inplace=True)
            st.success("Outliers removed")
            
    ###########################################################################
    # Step 9: Data validation
    ###########################################################################
    elif step == "Data Validation":  
        valid_age = (df_main["Age"] >= 0) & (df_main["Age"] <= 120)
        invalid_ages = df_main[~valid_age]
        st.write(f"Found {len(invalid_ages)} invalid age values.")
        if len(invalid_ages) > 0:
            st.dataframe(invalid_ages)
            fix_ages = st.checkbox("Set invalid ages to NaN?")
            if fix_ages:
                df_main.loc[~valid_age, "Age"] = np.nan
                st.success("Invalid ages set to NaN.")


########################
# End of if/else for data load
########################

else:
    st.warning("Please upload a CSV file or use the sample dataset to proceed.")

########################
# Final Notes / References
########################
st.header("Further Reading / References")
st.write("""
1. **Data Cleaning**: 
   - Van den Broeck et al. Data cleaning: detecting, diagnosing, and editing data abnormalities. 
     _PLoS Medicine_, 2005.
2. **Tidy Data**: 
   - Hadley Wickham. Tidy Data. _Journal of Statistical Software_, 2014.
3. **Missing Data**: 
   - Little & Rubin. _Statistical Analysis with Missing Data_, 3rd ed.
4. **Data Dictionaries & Documentation**: 
   - CDC Field Epidemiology Manual: [Data collection & management principles](https://www.cdc.gov/eis/field-epi-manual/index.html).
5. **Merging & Linking**: 
   - Practical tips in *Long & Freese, "Regression Models for Categorical Dependent Variables Using Stata"*, 
     chap. on data management.
6. **Duplicates & Data Validation**:
   - WHO. "Data Quality Review: A toolkit for facility data quality assessment." 
     (Sections on verifying data consistency, removing duplicates).

7. **Outlier Detection**:
   - Barnett V, Lewis T. _Outliers in Statistical Data_. Wiley, 1994.
   - Tukey JW. "Exploratory Data Analysis." Addison–Wesley, 1977 (for boxplots).

8. **Column Renaming & Type Conversion**:
   - Wickham H, Grolemund G. _R for Data Science_, O'Reilly (Chapters on data wrangling 
     concepts that also apply in Python/pandas).    
""")

st.header("Check your understanding")

# Quiz for Data Preview & Basic Summaries
if step == "Data Preview & Basic Summaries":
    q1 = st.radio(
        "What should be the **first step** when working with a new dataset?",
        [
            "Immediately run statistical tests",
            "Check the first few rows and data types",
            "Randomly delete some rows to clean the data",
            "Ignore missing values and move to analysis"
        ]
    )
    if q1 == "Check the first few rows and data types":
        st.success("✅ Correct! Always start by inspecting the dataset to understand its structure.")
    else:
        st.error("❌ Not quite. The correct first step is to explore the dataset with `.head()` and `.info()`.")

# Quiz for Missing Data Exploration
elif step == "Missing Data Exploration":
    q2 = st.radio(
        "Which of the following **is NOT** a common way to handle missing data?",
        [
            "Dropping rows with missing values",
            "Replacing missing values with a constant (e.g., mean, median)",
            "Ignoring missing values completely without any investigation",
            "Using advanced imputation techniques"
        ]
    )
    if q2 == "Ignoring missing values completely without any investigation":
        st.success("✅ Correct! You should never ignore missing values without checking their pattern.")
    else:
        st.error("❌ Not quite. **Ignoring missing values** without investigating them can introduce bias.")

# Quiz for Recoding / Creating New Variables
elif step == "Recoding / Creating New Variables":
    q3 = st.radio(
        "When **categorizing continuous variables** (e.g., Age into groups), which method is commonly used?",
        [
            "Randomly assigning category labels",
            "Using `pd.cut()` or `pd.qcut()` in Pandas",
            "Multiplying all values by a constant",
            "Changing numbers into strings"
        ]
    )
    if q3 == "Using `pd.cut()` or `pd.qcut()` in Pandas":
        st.success("✅ Correct! `pd.cut()` creates fixed bins, while `pd.qcut()` creates quantile-based bins.")
    else:
        st.error("❌ Not quite. Use **Pandas `cut` or `qcut`** to categorize continuous variables.")

# Quiz for Reshaping (Wide ↔ Long)
elif step == "Reshaping (Wide ↔ Long)":
    q4 = st.radio(
        "What is the **'wide' format** in data reshaping?",
        [
            "Each row represents one observation, and repeated measures are in separate columns",
            "Each row represents a single measurement",
            "Data is stored as a dictionary",
            "Wide format doesn't exist in data science"
        ]
    )
    if q4 == "Each row represents one observation, and repeated measures are in separate columns":
        st.success("✅ Correct! In **wide format**, repeated measurements appear as multiple columns.")
    else:
        st.error("❌ Not quite. **Wide format** means multiple columns for repeated measures.")

# Quiz for Merging Datasets
elif step == "Merging Datasets":
    q5 = st.radio(
        "Which type of **merge** keeps only rows that exist in **both datasets**?",
        [
            "Left Join",
            "Right Join",
            "Inner Join",
            "Outer Join"
        ]
    )
    if q5 == "Inner Join":
        st.success("✅ Correct! **Inner join** keeps only the rows that match in both datasets.")
    else:
        st.error("❌ Not quite. **Inner join** keeps only rows that are present in **both datasets**.")

# Quiz for Check / Remove Duplicates
elif step == "Check / Remove Duplicates":
    q6 = st.radio(
        "Which Pandas function is used to **identify duplicate rows**?",
        [
            "df.isna()",
            "df.dropna()",
            "df.duplicated()",
            "df.sort_values()"
        ]
    )
    if q6 == "df.duplicated()":
        st.success("✅ Correct! `df.duplicated()` flags duplicate rows.")
    else:
        st.error("❌ Not quite. Use **df.duplicated()** to find duplicates in the dataset.")

# Quiz for Renaming & Type Conversion
elif step == "Renaming & Type Conversion":
    q7 = st.radio(
        "Which function is used to **rename a column in Pandas**?",
        [
            "`df.rename(columns={'old_name': 'new_name'})`",
            "`df.columns = 'new_name'`",
            "`df.replace('old_name', 'new_name')`",
            "`df.convert('old_name', 'new_name')`"
        ]
    )
    if q7 == "`df.rename(columns={'old_name': 'new_name'})`":
        st.success("✅ Correct! Use `.rename(columns={})` to change column names in Pandas.")
    else:
        st.error("❌ Not quite. The correct function is `df.rename(columns={'old_name': 'new_name'})`.")

# Quiz for Outlier Detection
elif step == "Outlier Detection":
    q8 = st.radio(
        "What is a **common method** to detect outliers in numeric data?",
        [
            "Using mean and standard deviation",
            "Using a histogram",
            "Using a boxplot",
            "All of the above"
        ]
    )
    if q8 == "All of the above":
        st.success("✅ Correct! **Mean & SD, histograms, and boxplots** all help detect outliers.")
    else:
        st.error("❌ Not quite. **Outliers can be detected using multiple statistical methods.**")

# Quiz for Data Validation
elif step == "Data Validation":
    q9 = st.radio(
        "Which is an example of **data validation**?",
        [
            "Checking if Age values are within a valid range (e.g., 0-120)",
            "Randomly changing values",
            "Deleting columns with missing values without checking",
            "Saving the dataset without looking at it"
        ]
    )
    if q9 == "Checking if Age values are within a valid range (e.g., 0-120)":
        st.success("✅ Correct! Data validation ensures values make sense and are within expected ranges.")
    else:
        st.error("❌ Not quite. **Data validation involves checking for consistency and logical errors.**")

st.write("🎉 Great job! Keep practicing to master **Data Management & Wrangling** in Epidemiology. 🚀")


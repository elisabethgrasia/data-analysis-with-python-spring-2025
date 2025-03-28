### **Week 1: Introduction to Data Analysis & Practicalities**  

- **Objective**: Set up the environment and understand the data lifecycle.  
- **Tools**: Python, Visual Stuido Code, Jupyter Notebook on Anaconda, Google Colab.  
- **Topics**:  
  - What is data analysis? (Descriptive vs. diagnostic vs. predictive).  
  - Installing Python libraries (`pip`, `conda`).  
  - Data types (structured, semi struncture, unstructured data).  
  - Introduction to datasets (CSV, Excel, SQL).  
  - Common File fomats of datasets(.txt, .csv, .tsv, .json. .xml, .xlsx)
  - Ethics in data handling (GDPR, privacy).  
  - **Hands-on**: Loading a dataset and basic exploration.

---

### **Week 2: NumPy Fundamentals**  

- **Objective**: Master array operations for numerical computing.  
- **Tools**: NumPy.  
- **Topics**:  
  - Creating arrays (1D, 2D, 3D).  
  - Array operations (reshaping, slicing, broadcasting).  
  - Mathematical functions (aggregations, linear algebra).  
  - Random sampling (normal, uniform distributions).  
  - **Hands-on**: Solving numerical problems (e.g., Random data generation, matrix multiplication, ).

---

### **Week 3: Data Visualization with Matplotlib**  

- **Objective**: Create static, interactive, and publication-quality plots.  
- **Tools**: Matplotlib, Seaborn (optional).  
- **Topics**:  
  - Line plots, bar charts, histograms, scatter plots.  
  - Customizing plots (labels, legends, colors).  
  - Subplots and multi-panel figures.  
  - Introduction to Seaborn for statistical visuals.  
  - **Hands-on**: Visualizing different datasets

---

### **Week 4: Pandas for Data Manipulation**  

- **Objective**: Clean, transform, and analyze tabular data.  
- **Tools**: Pandas.  
- **Topics**:  
  - DataFrames vs. Series.  
  - Indexing (`loc`, `iloc`), filtering, and grouping.  
  - Handling missing data (`dropna`, `fillna`).  
  - Merging/joining datasets.  
  - **Hands-on**: Cleaning, Exploring, transforming, visualizing and  analysing different datasets

---

### **Week 5: Descriptive Statistics**  

- **Objective**: Summarize and interpret data distributions.  
- **Tools**: Pandas, SciPy.  
- **Topics**:  
  - Measures of central tendency (mean, median, mode).  
  - Measures of spread (variance, standard deviation, IQR).  
  - Skewness, kurtosis, and distributions (normal, binomial).  
  - Correlation and covariance.  
  - **Hands-on**: Exporing and analyzing a dataset.

---

### Exercise

#### **Task 1: Employee Dataset Analysis**  

**Objective**: Use Python and pandas to analyze the [Employee Dataset](https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset) and derive actionable insights.  You can download the [Employee Dataset](https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset) data from Kaggle, you need to create an account on Kaggle since it requires to  downolad datasets

**Requirements**:  

1. **Data Preparation**:  
   - Clean the dataset (handle missing values, duplicates, data types).  
   - Validate columns like `salary`, `age`, and `management` for consistency.  

2. **Exploratory Analysis**:  
   - Generate summary statistics (mean, median, distributions).  
   - Explore relationships between variables (e.g., `salary` vs. `education`, `management` vs. `environment` satisfaction).  

3. **Visualization**:  
   - Create visualizations (e.g., boxplots for salary distribution by education level, heatmaps for correlation analysis).  
   - Highlight trends (e.g., attrition patterns linked to `management` scores).  

4. **Key Questions**:  
   - Does higher education correlate with salary or job retention?  
   - Are there gender disparities in salary or promotion?  
   - What workplace factors (e.g., `environment`, `colleagues`) most impact employee satisfaction?  

---

#### **Task 2: Cat Breed API to CSV Transformation**  

**Objective**: Fetch data from [The Cat API](https://api.thecatapi.com/v1/breeds) and transform it into a structured `cats.csv` file.  

**Requirements**:  

1. **API Data Extraction**:  
   - Fetch breed data programmatically

2. **Data Transformation**:  
   - Map API fields to CSV headers:  

     ```csv
     ID, Name, Origin, Description, Temperament, Life Span (years), Weight (kg), Image URL
     ```  

   - **Special Cases**:  
     - Combine `temperament` as a comma-separated string (e.g., "Active, Curious").  
     - Convert `weight` from imperial to metric if necessary.  
     - Extract the first image URL from the breed’s `image` object.  

3. **Validation**:  
   - Handle missing fields (e.g., default `Description` to "N/A" if empty).  
   - Ensure numeric columns (`Life Span`, `Weight`) are properly formatted.  

    **Sample CSV Row**:  

    ```csv
    ID, Name, Origin, Description, Temperament, Life Span (years), Weight (kg), Image URL
    abys,Abyssinian,Egypt,"The Abyssinian is easy to care for...","Active, Energetic, Independent",14.5,4,https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg
    ```  

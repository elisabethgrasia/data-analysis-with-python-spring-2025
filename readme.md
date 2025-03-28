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
  - **Hands-on**: Solving numerical problems (e.g., matrix multiplication, simulations).

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
  - **Hands-on**: Cleaning a messy dataset (e.g., customer records).

---

### **Week 5: Descriptive Statistics**  

- **Objective**: Summarize and interpret data distributions.  
- **Tools**: Pandas, SciPy.  
- **Topics**:  
  - Measures of central tendency (mean, median, mode).  
  - Measures of spread (variance, standard deviation, IQR).  
  - Skewness, kurtosis, and distributions (normal, binomial).  
  - Correlation and covariance.  
  - **Hands-on**: Analyzing a dataset (e.g., GDP vs. population trends).

### Exercise

1. Flow [this example](https://nbviewer.org/github/juhanurmonen/data-analytics-basics-prepare-data/blob/main/Python_pandas_introduction.ipynb) to work on this data set [Employee dataset](https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset)

2. Fetch the cats breed from the cat's api and transform the data to cats.csv. Here is the link for the API(<https://api.thecatapi.com/v1/breeds>)

Heading of the CSV data with sample data looks like this:

```sh
ID,Name,Origin,Description,Temperament,Life Span (years),Weight (kg),Image URL
abys,Abyssinian,Egypt,"The Abyssinian is easy to care for...",Active,Energetic,Independent,Intelligent,Gentle,14.5,4,https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg
```

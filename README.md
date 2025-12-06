# Problem-Statement---2-HR-Data-Cleaning-for-Analysis
This project automates the cleaning and preprocessing of raw HR employee data using Python.
It transforms inconsistent, incomplete CSV files into a fully clean, analysis-ready dataset suitable for:

Exploratory Data Analysis (EDA)

Statistical insights

Machine Learning models

HR analytics dashboards

🚀 Project Overview

Real-world HR datasets often suffer from:

Missing values

Incorrect data types

Inconsistent categorical labels

Unstructured or noisy data

This script provides a reliable data-cleaning pipeline that:

✔ Handles missing values intelligently
✔ Corrects incorrect and inconsistent data types
✔ Encodes categorical features
✔ Outputs a fully processed DataFrame

The entire process is wrapped inside a single function called clean_hr_data() for reuse and automation.

🧠 Key Features
🔹 1. Missing Value Handling

Numerical columns → filled using median

Categorical columns → filled using mode
This ensures stability and prevents data skew.

🔹 2. Data Type Standardization

Age, Salary, Experience → converted to integers

EmployeeID → converted to string

🔹 3. Categorical Encoding

One-hot encoding applied using pd.get_dummies()

drop_first=True used to prevent the dummy variable trap

🔹 4. Clean, Ready-to-Use Dataset

Final output is perfect for use in ML pipelines, visualizations, and dashboards.


The function:

Reads the raw CSV

Displays initial data info

Fills missing values

Fixes incorrect data types

Encodes categorical variables

Returns a clean DataFrame
📂 Project Structure
.
├── dummy_hr_data.csv        # Example dataset
├── hr_cleaning_script.py    # Main data-cleaning script
└── README.md                # Project documentation

🧩 How the Function Works
cleaned_df = clean_hr_data("filename.csv")

🛠️ Technologies Used

Python

Pandas

NumPy

📘 Example Output

✔ Zero missing values
✔ All data types corrected
✔ Encoded categorical variables
✔ Clean DataFrame ready for ML

The script prints:

Before & after missing value summary

Type corrections

Encoded columns

Head of the clean DataFrame

📊 Use Cases

This cleaned HR dataset can be used for:

Employee churn prediction

Performance analysis

Salary modelling

HR dashboards

Attrition analytics

📞 Author

Shashi Kumar
B.Sc. Computer Science & Data Analytics
Indian Institute of Technology Patna

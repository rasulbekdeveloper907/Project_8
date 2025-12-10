# 🚗 Car Kilometer Prediction Project

## 📌 Project Overview
This project focuses on predicting the **mileage (`kilometer`)** of cars based on various features. The dataset contains **371,528 entries** with **21 columns**, including both numerical and categorical attributes. The target variable for this regression task is `kilometer`.

The main goal is to build a machine learning model that can accurately predict the mileage of a vehicle given its characteristics.

---

## 🗂️ Dataset Description

| Column                 | Description |
|------------------------|-------------|
| `index`                | Unique index for each entry |
| `dateCrawled`          | Date when the listing was crawled |
| `name`                 | Car listing name |
| `vehicleType`          | Type of vehicle (e.g., sedan, SUV) |
| `yearOfRegistration`   | Year when the car was registered |
| `model`                | Car model |
| `monthOfRegistration`  | Month when the car was registered |
| `fuelType`             | Type of fuel (e.g., petrol, diesel) |
| `brand`                | Brand of the car |
| `postalCode`           | Postal code of the seller |
| `lastSeen`             | Last seen timestamp of the listing |
| `abtest_control`       | A/B test control group indicator |
| `gearbox_automatik`    | Gearbox type (automatic or manual) |
| `notRepairedDamage_ja` | Indicates whether car had unrepaired damage |
| `kilometer_cont`       | Mileage of the car (target variable) |

---

## 🛠️ Project Workflow

1. **🧹 Data Cleaning & Preprocessing**
   - Handle missing values in both numerical and categorical columns.
   - Fill numerical missing values using **mean** and categorical missing values using **mode**.
   - Encode categorical variables:
     - One-hot encoding for columns with low cardinality 🎯.
     - Label encoding for columns with high cardinality 🏷️.
   - Scale numerical features (excluding the target `kilometer`) using **MinMaxScaler** 📏.

2. **📊 Exploratory Data Analysis (EDA)**
   - Understand distributions of numerical and categorical features 📈.
   - Visualize relationships between features and target variable (`kilometer`) 🔍.
   - Check for outliers and data inconsistencies ⚠️.

3. **🤖 Modeling**
   - Apply regression algorithms to predict `kilometer`.
   - Potential models include:
     - Linear Regression 📐
     - Random Forest Regressor 🌳
     - Gradient Boosting Regressor 🚀
     - XGBoost / LightGBM ⚡
   - Evaluate model performance using metrics such as **MAE, MSE, RMSE, and R² score** 📏.

4. **🚀 Deployment (Optional)**
   - Prepare the pipeline for future deployment.
   - Include preprocessing steps so that new data can be used for prediction.

---

## 🧰 Technologies & Libraries
- **Python 3.x 🐍**
- **Pandas 🐼**
- **NumPy 🔢**
- **scikit-learn 🎓**
- **Matplotlib / Seaborn / Plotly 📊**
- **Git / GitHub 🗃️**
- **Git LFS 🗄️**

---

## 🎯 Target Variable
- `kilometer`: Represents the mileage of the car 🛣️.
- Regression task: Predict numerical value based on other car features.

---

## 📁 File Structure

Project_9/
│
├── Scripts/
│ ├── data_preprosessing.py # Preprocessing classes: MissingValueImputer, Encoder, Scaler
│
├── Data/
│ ├── Raw_Data/ # Original dataset files
│ └── Preprocessed/ # Processed dataset files
│
├── Notebooks/ # Jupyter notebooks for EDA and experiments
│
└── README.md # Project overview and instructions


---

## ⚡ How to Use

1. Clone the repository:
```bash
git clone https://github.com/rasulbekdeveloper907/Project_9.git


📜 License

This project is for educational purposes. Dataset usage may be subject to its original license.

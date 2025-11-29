# 🚗 Car Dataset Regression Project

## 📌 Project Overview
This project focuses on predicting the **mileage (`kilometer`)** of cars based on various features. The dataset contains **371,528 entries** with **21 columns**, including both numerical and categorical attributes. The target variable for this regression task is `kilometer`.

The main goal is to build a machine learning model that can accurately predict the mileage of a vehicle given its characteristics.

---

## 🗂️ Dataset Description

| Column Name           | Data Type | Description |
|-----------------------|-----------|-------------|
| index                 | int64     | Row index |
| dateCrawled           | object    | Date the data was crawled |
| name                  | object    | Car name/title |
| seller                | object    | Seller type |
| offerType             | object    | Offer type |
| price                 | int64     | Price of the car 💰 |
| abtest                | object    | A/B test group 🧪 |
| vehicleType           | object    | Vehicle type 🚙 |
| yearOfRegistration    | int64     | Year of registration 📅 |
| gearbox               | object    | Gearbox type ⚙️ |
| powerPS               | int64     | Horsepower 🐎 |
| model                 | object    | Car model 🚘 |
| kilometer             | int64     | **Target: Mileage of the car 🛣️** |
| monthOfRegistration   | int64     | Month of registration 📆 |
| fuelType              | object    | Fuel type ⛽ |
| brand                 | object    | Car brand 🏷️ |
| notRepairedDamage     | object    | Repair status 🔧 |
| dateCreated           | object    | Date the ad was created 📄 |
| nrOfPictures          | int64     | Number of pictures in the ad 📷 |
| postalCode            | int64     | Postal code of the seller 📮 |
| lastSeen              | object    | Last seen date of the ad 👀 |

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
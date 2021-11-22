---
title: "Solution Approach - Analytics Vidhya November Jobathoon"
description: "Documentation of the solution approach for Employee Attrition Prediction Problem in Analytics Vidhya November Jobathon"
layout: post
toc: false
comments: true
image: images/vignette/AV-Nov.png
hide: false
search_exclude: false
categories: [markdown, jobathon, competetion, analytics]
---
<h1>Table of Contents</h1>

[1. Problem Statement](#1-problem-statement)
[2. Data Wrangling](#2-data-wrangling)
[3. Results from the EDA](#3-results-from-the-eda)
[4. Feature Engineering](#4-feature-engineering)
[5. Modeling](#5-modeling)      

# 1. Problem Statement
Develop a Machine Learning model to aid HR Department in predicting the attrition of employees.
<h2>Given</h2>

**Train Dataset** - Given with the following features:
1. Demographics of the employee 
2. Tenure information
3. Historical data regarding the performance

**Target** - There is no apparent target variable given in the dataset. The target is given in the column *`LastWorkingDate`* of the dataset.
$$
Target = \begin{cases}
   Date &\text{If } \ the\ employee\ did\ not\ leave\ the\ company \\
   Null &\text{If } \ the\ employee\ left\ the\ company
\end{cases}
$$

Filled the missing values in `LastWorkingDate` column with the `ReportingDate` column.

**Test Dataset** - Given only the Employee ID's for which we need to predict if they will leave the company or not in the *next two quarters of year 2018*.

**Evaluation Metric** - F1 Score

# 2. Data Wrangling
To make the data more suitable for Machine Learning models, EDA, and Feature Engineering, a few Data Wrangling steps were taken:
1. Creating the Target Variable from the `LastWorkingDate` column
2. Fill missing values in the `LastWorkingDate` column with the `ReportingDate` column
3. Convert all the Date columns to datetime format

# 3. Results from the EDA
Following are the  most prominent observations made from the EDA:
1. KDE Plots showed that the `Age` and `Salary` were normally distributed.
2. `Total Business Value` had a lot of zero values.
3. `Salary` and `Age` have a similar distribution for each `Gender`, `City` Category.
4. Employees with less Quarterly Rating tend to leave the company.
5. Older Employees have a much less probability of leaving the company, So `JoiningYear`, `Tenure` (in days, months, and years)  would be great features to use in the model.
6. Employees who did not leave the company have a much **higher number of Positive** `Total Business Value`.

# 4. Feature Engineering
**Features that were created based on EDA:**
1. `JoiningYear` - Year in which the employee joined the company.
2. `WorkingDays` - Number of working days till the reporting date.
3. `WorkingMonths` - Number of working months till the reporting date.
4. `WorkingYears` - Number of working years till the reporting date.
5. `Promotions` - Number of promotions till the reporting date.
6. `Quarterly_Rating_RA` - Running Average of the `QuarterlyRating`.
7. `Quarterly_Rating_CumSum` - Cumulative sum of the `QuarterlyRating`.
8. `Total_Business_Value_CumSum` - Cumulative sum of the `Total Business Value`.
9. `PBVCount` - Number of positive `Total Business Value` till the reporting date.
10. `NBVCount` - Number of negative `Total Business Value` till the reporting date.
11. `SalaryGrowth` - Growth in Salary till the reporting date.
12. `SalaryGrowthRatio` - Growth in Salary till the reporting date as a ratio.
13. `SalaryGrowth_WorkingDays` - Ratio of `SalaryGrowth` to `WorkingDays`
14. `SalaryGrowth_WorkingMonths` - Ratio of `SalaryGrowth` to `WorkingMonths`
15. `SalaryGrowth_WorkingYears` - Ratio of `SalaryGrowth` to `WorkingYears`
16. `Designation_Count` - Total number of employees with the same designation till the reporting date.
17. `ReportingDate_Count` - Total number of employees reported on a reporting date.
18. `City_ReprotingDate_Count` - Total number of employees reported in a particular city on a reporting date.
19. `City_Count` - Total number of employees in a particular city.

**Steps to assess the features:**
1. Calculate the `Mutual Information` between the features and the target variable.
2. Create a `Bar Plot` to show the `Mutual Information`.
3. Get 5-Fold Cross Validation Score for a CatBoost Base model (without Hyper Parameter Tuning).
4. Plot the `Feature Importance` of the CatBoost model.

**Discarded Features based on assessment:**
1. `SalaryGrowth_WorkingDays`
2. `SalaryGrowth_WorkingMonths`
3. `Designation_Count`
4. `SalaryGrowthRatio`
5. `SalaryGrowth`
6. `SalaryGrowth_WorkingYears`
7. `NBVCount`
8. `City_Count`

# 5. Modeling
**List of models used during the model-building:**
1. XGBoost
2. CatBoost
3. LightGBM
4. Random Forest
5. MLP (with EaryStopping and ReduceLROnPlateau)

**Results from the model building:**
1. Used 5-Fold StratifiedKFold cross validation to assess the model at every step.
2. MLP did not perform well due to less data available for training, and quickly overfit.
3. XGBoost, LightGBM, and Radom Forest performed worse than the CatBoost model.
4. Hyper Parameter Tuning was done for all the models using Optuna.
5. All models were trained using Early Stopping Rounds.
6. Models other than CatBoost did not improve the score even after Hyper Parameter Tuning.
7. CatBoost model was the best performing model.

<h2>Final Model</h2>

1. Ensemble of `14 CatBoost Models` with different Hyper Parameters and Random SEEDs.
2. Average of 5-Fold out-of-fold scored predictions from each model to create `Meta Features`.
3. Trained a Logistic Regression model on the meta features.
4. Final Inference on the Test Dataset - Using the Logistic Regression model.
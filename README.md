
# Churn Prediction Challenge

Customer churn is when customers stop using a company's product or service. Churn prediction identifies these at-risk customers, allowing businesses to take proactive measures to retain them. In this challenge, you'll predict churn for a video streaming service. Using customer data, you will build a machine learning model to determine whether a subscriber will continue their subscription for another month. Accurate churn predictions help businesses implement targeted retention strategies, ensuring customer satisfaction and loyalty.

## Table of Contents
- [Introduction](#introduction)
- [Data Description](#data-description)
- [Problem Statement](#problem-statement)
- [Requirements](#requirements)
- [Workflow](#workflow)
- [Results](#results)
- [Submission](#submission)
- [Example Code](#example-code)

## Introduction
Subscription services are widely used across various industries, including fitness, video streaming, and retail. One of the key objectives for these companies is to reduce churn and retain subscribers. Machine learning models can help predict which users are at risk of churning, enabling companies to take proactive measures.

In this challenge, you are tasked with building a model to predict whether subscribers of a video streaming service will continue their subscriptions for another month.

## Data Description
The datasets provided for this challenge are `train.csv` and `test.csv`. Each dataset contains information about subscribers, their preferences, and activity on the streaming platform.

- `train.csv`: Contains 243,787 samples with 21 columns, including the target variable `Churn`.
- `test.csv`: Contains 104,480 samples with the same features as `train.csv` but without the `Churn` column.

### Features
- `AccountAge`: The age of the user's account in months.
- `MonthlyCharges`: The amount charged to the user on a monthly basis.
- `TotalCharges`: The total charges incurred by the user over the account's lifetime.
- `SubscriptionType`: The type of subscription chosen by the user (Basic, Standard, or Premium).
- `PaymentMethod`: The method of payment used by the user.
- `PaperlessBilling`: Indicates whether the user has opted for paperless billing (Yes or No).
- `ContentType`: The type of content preferred by the user (Movies, TV Shows, or Both).
- `MultiDeviceAccess`: Indicates whether the user has access to the service on multiple devices (Yes or No).
- `DeviceRegistered`: The type of device registered by the user (TV, Mobile, Tablet, or Computer).
- `ViewingHoursPerWeek`: The number of hours the user spends watching content per week.
- `AverageViewingDuration`: The average duration of each viewing session in minutes.
- `ContentDownloadsPerMonth`: The number of content downloads by the user per month.
- `GenrePreference`: The preferred genre of content chosen by the user.
- `UserRating`: The user's rating for the service on a scale of 1 to 5.
- `SupportTicketsPerMonth`: The number of support tickets raised by the user per month.
- `Gender`: The gender of the user (Male or Female).
- `WatchlistSize`: The number of items in the user's watchlist.
- `ParentalControl`: Indicates whether parental control is enabled for the user (Yes or No).
- `SubtitlesEnabled`: Indicates whether subtitles are enabled for the user (Yes or No).
- `CustomerID`: A unique identifier for each customer.
- `Churn`: The target variable indicating whether a user has churned or not (1 for churned, 0 for not churned).

## Problem Statement
Your goal is to build a machine learning model that predicts the likelihood of each subscriber churning (canceling their subscription) in the next month.

## Requirements
To complete this challenge, you will need the following:
- Python 3.x
- Jupyter Notebook or any Python IDE
- The following Python packages:
  - pandas
  - numpy
  - scikit-learn
  - imbalanced-learn (for handling imbalanced data)
  - xgboost (optional, for advanced modeling)
  - matplotlib (for visualization)
  - seaborn (for visualization)

## Workflow
1. **Data Loading and Preprocessing**:
    - Load the datasets (`train.csv` and `test.csv`).
    - Drop the `CustomerID` column as it is not needed for modeling.
    - Handle missing values if any.
    - Apply log transformation to skewed features like `TotalCharges`.
    - Encode categorical variables using one-hot encoding.
    - Scale the features using `StandardScaler`.

2. **Feature Engineering**:
    - Create interaction and polynomial features if necessary.
    - Identify and remove or transform outliers.

3. **Model Training**:
    - Split the training data into training and validation sets.
    - Train a logistic regression model using `LogisticRegression` from scikit-learn.
    - Use `GridSearchCV` to perform hyperparameter tuning.
    - Evaluate the model on the validation set using metrics like accuracy, ROC-AUC, and confusion matrix.

4. **Prediction**:
    - Use the trained model to predict churn probabilities on the test set.
    - Prepare the submission file in the required format.

5. **Submission**:
    - Create a DataFrame with `CustomerID` and `predicted_probability`.
    - Save the DataFrame as `submission.csv`.

## Results
- The model performance will be evaluated based on the ROC-AUC score on the test set.
- Use cross-validation to ensure the model generalizes well to unseen data.
- Got 80% percentile among the peers. 

## Submission
Submit the predictions as a CSV file with the following format:
- `CustomerID`: The unique identifier for each customer from the test set.
- `predicted_probability`: The probability of the customer churning.

# FINANCIAL INCLUSION IN AFRICA FOR BOTSWANA

## Introduction
This Python code is designed to train a machine learning model for a classification task and make predictions on a test dataset. It uses various libraries for data manipulation, visualization, and machine learning. The primary focus is on using the XGBoost algorithm for the classification task.

## How To Setup Folders
1. Download and extract the folder containing the solution and the other required files.
2. Open google collab and upload the solution1.ipynb file then open it.
3. Upload Test.csv, Train.csv, VariableDefinitions.csv and SampleSubmission.csv to the session storage.
4. Run the code.
5. After running the entire code, the "first_submission.csv" file will be added to the session strorage where it can be downloaded.

## Features Used in the Code
1. **Libraries**:
   - The code imports several Python libraries for data processing and machine learning, including `pandas`, `numpy`, `seaborn`, `matplotlib`, `LightGBM`, `sklearn`, `xgboost`, and more.

2. **Data Handling**:
   - The code loads training and test data from CSV files using `pd.read_csv`.
   - It preprocesses the data, which includes encoding categorical features and scaling numerical features.
   - The training data is split into training and validation sets using `train_test_split`.
   - An XGBoost classifier is trained on the data, and its performance is evaluated using accuracy and a confusion matrix.
   - Hyperparameter tuning is performed through a grid search using `GridSearchCV`.

## Environment
The code is designed for a Jupyter Notebook or a similar interactive environment. It utilizes specific imports like `from google.colab import files`, suggesting that it's meant to run in Google Colab. The environment should have Python and the required libraries installed.

## Hardware Requirements
The code does not have particularly high hardware requirements. It should run on standard laptops or desktops with Python and the necessary libraries. However, for larger datasets and during grid search, computational resources become more critical. Having a machine with a reasonable amount of RAM and a decent CPU is recommended. Additionally, if available, GPU acceleration can significantly speed up training, especially for XGBoost.

## Order of Execution
To successfully execute this code, follow these steps:

1. Import the required libraries.
2. Load the data from CSV files ('Train.csv', 'Test.csv', 'SampleSubmission.csv', 'VariableDefinitions.csv').
3. Preprocess the data using the `preprocessing_data` function.
4. Split the training data into training and validation sets with `train_test_split`.
5. Train an initial XGBoost model using the default hyperparameters.
6. Evaluate the initial model's performance with accuracy and a confusion matrix.
7. Perform hyperparameter tuning through grid search using `GridSearchCV`.
8. Train the best model found by grid search.
9. Make predictions on the test dataset.
10. Create a submission file for the competition.
11. Save the submission file as "first_submission.csv".

Ensure that you have the necessary data files in the same directory as your Jupyter Notebook or specify the correct file paths. Also, make sure to have the required Python libraries installed, particularly XGBoost, LightGBM, pandas, scikit-learn, and others as necessary.

## Expected Run Time 
The expected run time for this notebook is approximately 1 minute and 30 seconds.

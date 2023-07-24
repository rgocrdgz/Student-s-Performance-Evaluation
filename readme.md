## Classification Model Evaluation

This repository contains code for building and evaluating classification models on a given dataset. The code is written in Python and utilizes various machine learning algorithms and techniques.

### Dataset

The dataset used for the classification task is stored in the file data.csv. It is preprocessed to exclude certain categories and entries that are not relevant to the analysis.

### Dependencies

The code requires the following dependencies:

- numpy
- pandas
- seaborn
- matplotlib
- plotly
- scikit-learn

You can install these dependencies by running the following command:

pip install -r requirements.txt

### Code Structure

The code is organized as follows:

- functionality.py: This module contains helper functions used in the main code file.
- main.py: This is the main code file that performs the classification modeling and evaluation.
- code.py: This is the main code file that performs the classification modeling and evaluation in a jupyter notebook.


### Usage

To run the code and evaluate the classification models, follow these steps:

1. Install the required dependencies as mentioned in the Dependencies section.
2. Place the dataset file data.csv in the same directory as the code files.
3. Run the main.py file using the following command:

python main.py

(Or run the code.ipynb file.) 

4. The code will perform the following steps:
    - Load and preprocess the dataset.
    - Create the feature matrix and target variable.
    - Instantiate a list of classification models to be trained and compared.
    - Perform data transformation and feature engineering.
    - Split the data into training and validation sets.
    - Train and evaluate the models using cross-validation.
    - Plot the performance results for the training and validation sets.
    - Apply feature selection using variance threshold.
    - Train and evaluate the models with feature selection.
    - Perform grid search for hyperparameter tuning on selected models.
    - Train and test the best performing models on the training and test sets.
    - Plot the performance results for the training and test sets.
    - Select the best model based on performance metrics.
    - Train the best model on the entire training data.
    - Generate a classification report for the best model on the training and test sets.
5. The results and evaluation metrics will be displayed in the console, and relevant plots will be generated to visualize the performance of the models.# Student-s-Performance-Evaluation

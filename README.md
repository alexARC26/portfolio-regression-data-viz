# Portfolio: 1. CO2 Emissions: Analysis and Regression Prediction
Author: Alejandro Rodríguez-Collado

## Description
- **1.1. CO2 Emissions Data Profiling and Visualization**: Conducted exploratory data analysis and preprocessing on a dataset of 7,385 Canadian vehicles to predict CO2 emissions. The project involved data cleaning, handling duplicates, feature engineering (e.g., splitting the Transmission column), and visualizing numerical and categorical variables using boxplots, histograms, scatterplots, and correlation heatmaps. Identified high-pollution outliers and redundant variables, preparing the dataset for regression modeling in a subsequent phase.
- **1.2. CO2 Emissions Regression Modeling**: Description: Developed and evaluated regression models to predict CO2 emissions from a preprocessed dataset of 6,281 Canadian vehicles with 7 features (e.g., fuel consumption, engine size, cylinders). Implemented Linear Regression, LASSO Regression, Support Vector Machine (SVM) with RBF kernel, and Regression Trees. Evaluated models using $R^2$ and MSE, identifying SVM as the most robust ($R^2$: 0.9131 test, MSE: 299.94 test) for its balance of accuracy and generalization. Highlighted Fuel Consumption City as the most influential feature and noted Transmission’s limited impact.

## Dataset
Debajyoti Podder (2020). Dataset: CO2 Emission by Vehicles - Amount of CO2 emissions by a vehicle depending on their various features. URL: [https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)

## Techniques and Tools
- Techniques: Data cleaning, feature engineering, univariate and bivariate analysis, correlation analysis, data normalization, categorical encoding, linear regression, LASSO regularization, SVM, regression trees, train-test split, feature importance analysis, model evaluation.
- Tools: Python. Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (LinearRegression, SVR, DecisionTreeRegressor, StandardScaler, OrdinalEncoder), Statsmodels, Kagglehub.

## Main Results
- Preprocessed dataset from 7,385 to 6,281 vehicles, removing duplicates and irrelevant features (e.g., Make, Model), and engineered features like Gears from Transmission.
- Identified Fuel Consumption City (L/100 km) as the most influential predictor of CO2 emissions via correlation heatmaps and regression tree feature importance, with Transmission being statistically insignificant.
- Support Vector Machine (SVM) with RBF kernel achieved the best performance ($R^2$: 0.9131 test, MSE: 299.94 test), excelling in capturing non-linear patterns and generalizing well.
- Regression Trees showed near-perfect training accuracy ($R^2$: 0.9997) but slight overfitting ($R^2$: 0.9929 test, MSE: 24.55 test), suggesting potential for pruning or ensemble methods.

## Project Structure
- `notebooks/1-Data-Profiling-Visualization.ipynb`: 1.1. CO2 Emissions Data Profiling and Visualization.
- `notebooks/2-ML-Regression.ipynb`: 1.2. CO2 Emissions Regression Modeling.

## Execution
The notebooks are designed for seamless execution in Google Colab. They include integrated data downloads from Kaggle and all necessary dependencies, ensuring immediate reproducibility without additional setup. This approach streamlines the user experience, allowing direct focus on the data analysis and modeling workflows.
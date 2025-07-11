# Portfolio: CO2 Emissions: Analysis and Regression Prediction
This project develops a model to predict CO2 emissions of cars using the Canadian Vehicles dataset. An exploratory data analysis is conducted, followed by preprocessing, including data cleaning, handling duplicates, feature engineering, and selection. The preprocessed dataset (6,281 cars, 7 features) is used to train several ML models: linear regression, LASSO regression, support vector machine (SVM), regression trees, and XGBoost. The latter model, XGBoost, achieves an $R^2$ of **0.998** on an independent test dataset, enabling applications like automotive design (key features of eco-friendly vehicles) or consumer tools for sustainable purchasing decisions.

## Dataset and Preprocessing
- **Dataset**: 7,385 vehicles, 6 features, and 1 response variable from Debajyoti Podder (2020). Dataset: CO2 Emission by Vehicles - Amount of CO2 emissions by a vehicle depending on its various features. URL: [https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)
- **Preprocessing**: Reduced to 6,281 vehicles, 6 features and 1 response variable after removing duplicate rows, irrelevant features, and engineering new features like gears and transmission. Categorical variables were recoded into numerical values, and numerical variables were scaled.

## Notebooks
- [Data Profiling, Transformation, and Visualization](https://github.com/alexARC26/portfolio-regression-data-viz/blob/main/notebooks/1_Profiling_Visualization.ipynb)
- [Machine Learning - Regression models](https://github.com/alexARC26/portfolio-regression-data-viz/blob/main/notebooks/2_ML_Regression.ipynb)

## Methodology
For Data Profiling and Visualization, several univariate, bivariate, and multivariate analyses were conducted using basic statistics (mean, standard deviation, percentiles, correlation, absolute and relative frequencies) as well as classic visualizations (boxplot, bar chart, pie chart, scatterplot, heatmaps).

For the Regression notebook, the dataset was split into train (2/3) and test (1/3) sets. Five models were evaluated: linear regression, LASSO regression, SVM with RBF kernel, regression trees, and XGBoost. Hyperparameters were tuned using cross-validation with the train set.

The notebooks are designed for seamless execution in Google Colab, including integrated data downloads from Kaggle and all necessary dependencies, ensuring immediate reproducibility without additional setup.

## Results
The Data Profiling and Visualization provided several compelling insights, including duplicate and outlier detection (e.g., high-pollution Bugatti cars), redundant or purposeless variables in ML uncovered (including the categorical variable vehicle class and combined consumption variables), feature engineering (addition of gears and transmission features), and high correlations uncovered (as a car's engine size and cylinder count increase, fuel consumption and pollution rise).

The regression models were evaluated on the test set using $R^2$ and mean squared error. The bar chart below compares all five models, with the **XGBoost** achieving the best performance: **0.998 $R^2$ and 7.400 MSE**. Other notable results include regression trees at 0.993 $R^2$ and SVM with RBF kernel at 0.913 $R^2$. Fuel consumption in city was identified as a key predictor using linear model p-values and tree-based feature importance metrics.
![Model Performance by Metric](https://raw.githubusercontent.com/alexARC26/portfolio-regression-data-viz/main/images/Results_Summary.png)

## Technologies Used
- Data exploration and transformation: `numpy` and `pandas`.
- ML / AI: `scikit-learn` for ML models (SVR, DecisionTreeRegressor) and evaluation metrics, `statsmodels` for linear regression and `xgboost` for the XGBoost implementation.
- Visualization: `matplotlib` and `seaborn`.

## Challenges and Solutions
- **Poor Feature Selection**: The original dataset included redundant variables and under-used information.
  - **Solution**: Redundant variables identified via correlation heatmaps; feature engineering added gears and transmission variables.
- **Raw Dataset for ML**: Data included duplicates or strictly different cars (e.g., one vehicle using natural gas).
  - **Solution**: The aforementioned rows discarded by identifying them with statistics, boxplots, and scatterplots.
- **Non-linear Relation**: Pollution has no direct linear relation with the provided feature set.
  - **Solution**: Introduced non-linear methods like SVM with RBF kernel, regression trees, and XGBoost.
- **Overfitting**: Models mislabeled unseen data.
  - **Solution**: Used robust models like SVM; tuned XGBoost hyperparameters (e.g., tree depth, regularization).

## Future Work
- Expand dataset with additional cars and features; trained models may have limited generalizability due to regional biases.
- More feature engineering, incorporating features like vehicle weight or driving conditions.
- Further optimization of hyperparameters (e.g., dropout rates, filter sizes, number of layers) for better generalization; apply to SVM and regression tree models.
- Explore additional ensemble methods, such as stacking XGBoost with SVM or Random Forests.
- Model interpretability insights for the more complex models.

## Explore More
Check out my full portfolio at [GitHub](https://github.com/alexARC26) or connect via [LinkedIn](https://www.linkedin.com/in/alejandro-rodr%C3%ADguez-collado-a3456b17a) to discuss ML projects!
# Portfolio: CO2 Emissions: Analysis and Regression Prediction
This project develops a predictive model for car CO2 emissions using the Canadian Vehicles dataset. It begins with exploratory data analysis, followed by preprocessing steps, including data cleaning, duplicate removal, feature engineering, and feature selection. The refined dataset, comprising 6,281 vehicles and 7 key features, is used to train multiple machine learning models: linear regression, LASSO regression, support vector machine (SVM), regression trees, and XGBoost. The XGBoost model achieves an exceptional $R^2$ of **0.998** on an independent test dataset, enabling applications such as identifying key features for eco-friendly vehicle design and supporting consumer tools for sustainable purchasing decisions.

## Dataset and Preprocessing
- **Dataset**: 7,385 vehicles, 6 features, and 1 response variable from Debajyoti Podder (2020). Dataset: CO2 Emission by Vehicles - Amount of CO2 emissions by a vehicle depending on its various features. URL: [https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)
- **Preprocessing**: The dataset was reduced to 6,281 vehicles with 6 features and 1 response variable after eliminating duplicates, removing irrelevant features, and engineering new features, such as gear count and transmission type. Categorical variables were converted to numerical values, and numerical variables were standardized through scaling.

## Notebooks
- [Data Profiling, Transformation, and Visualization](https://github.com/alexARC26/portfolio-regression-data-viz/blob/main/notebooks/1_Profiling_Visualization.ipynb)
- [Machine Learning - Regression models](https://github.com/alexARC26/portfolio-regression-data-viz/blob/main/notebooks/2_ML_Regression.ipynb)

## Methodology
Univariate, bivariate, and multivariate analyses were performed using descriptive statistics (mean, standard deviation, percentiles, correlations, and frequency distributions) and visualizations, including boxplots, bar charts, pie charts, scatterplots, and heatmaps, to explore the dataset comprehensively.

The dataset was divided into training (2/3) and test (1/3) sets. Five models —linear regression, LASSO regression, SVM with RBF kernel, regression trees, and XGBoost— were evaluated. Hyperparameters were optimized through cross-validation on the training set.

The notebooks are designed for seamless execution in Google Colab, including integrated data downloads from Kaggle and all necessary dependencies, ensuring immediate reproducibility without additional setup.

## Results
Data Profiling and Visualization revealed compelling insights, including the detection of duplicates and outliers (e.g., high-pollution Bugatti vehicles), identification of redundant or irrelevant variables for machine learning (e.g., the categorical variable "vehicle class" and combined consumption metrics), feature engineering (e.g., adding gear count and transmission features), and strong correlations (e.g., larger engine size and cylinder count correlating with higher fuel consumption and emissions).

The regression models were assessed on the test set using $R^2$ and mean squared error (MSE). The bar chart below compares the five models, with XGBoost achieving the highest performance: **0.998 $R^2$ and 7.400 MSE**. Other notable results include regression trees with 0.993 $R^2$ and SVM with RBF kernel at 0.913 $R^2$. Fuel consumption in city emerged as a key predictor, based on linear model p-values and tree-based feature importance metrics.

![Model Performance by Metric](https://raw.githubusercontent.com/alexARC26/portfolio-regression-data-viz/main/images/Results_Summary.png)
*Figure 1: Model performance for CO₂ emissions prediction, evaluated by R² and MSE.*

## Technologies Used
- Data exploration and transformation: `numpy` and `pandas`.
- ML / AI: `scikit-learn` for ML models (SVR, DecisionTreeRegressor) and evaluation metrics, `statsmodels` for linear regression and `xgboost` for the XGBoost implementation.
- Visualization: `matplotlib` and `seaborn`.

## Challenges and Solutions
- **Poor Feature Selection**: The original dataset included redundant variables and underutilized information.
  - **Solution**: Identified redundant variables using correlation heatmaps and introduced feature engineering by adding gear count and transmission variables.
- **Raw Dataset for ML**: The dataset included duplicates and unique outliers (e.g., a vehicle using natural gas).
  - **Solution**: Removed these rows after detection through statistical analysis, boxplots, and scatterplots.
- **Non-linear Relation**: Pollution lacked a direct linear relationship with the provided features.
  - **Solution**: Employed non-linear models, including SVM with RBF kernel, regression trees, and XGBoost.
- **Overfitting**: Models struggled to generalize to unseen data.
  - **Solution**: Utilized robust models like SVM and tuned XGBoost hyperparameters (e.g., tree depth and regularization) to mitigate overfitting.

## Future Work
- Expand dataset with diverse vehicles (to reduce regional biases) and additional features (e.g., engine types, fuel efficiency).
- Add features like vehicle weight or driving conditions for improved model performance.
- Further optimization of hyperparameters for better generalization. Tune SVM and regression tree models.
- Explore ensemble methods like stacking XGBoost with SVM or Random Forests.
- Enhance interpretability of complex models using SHAP or LIME.

## Explore More
Check out my full portfolio at [GitHub](https://github.com/alexARC26) or connect via [LinkedIn](https://www.linkedin.com/in/alejandro-rodr%C3%ADguez-collado-a3456b17a) to discuss ML projects!
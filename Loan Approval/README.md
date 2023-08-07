# 💳 Loan Approval Project

**Overview:** <br>
This project centres on identifying potential defaulters for a bank that provides loan services. The primary objective is to leverage historical loan data of customers and accurately pinpoint individuals at high risk of defaulting on their loans. The ultimate goal is to enhance the bank's loan approval process by potentially excluding those identified as high-risk default candidates from accessing loan services.

**Skills and Tools Used:** <br>
The Pandas and NumPy libraries facilitated data manipulation and preprocessing, while the Matplotlib and Seaborn libraries enabled insightful visualizations. The application of TargetEncoder and StandardScaler from scikit-learn ensured effective feature engineering and scaling. The predictive modeling phase made use of an ensemble of classifiers. The models were evaluated using performance metrics from scikit-learn as well.

**Approach and Techniques:** <br>
The project's approach encompassed strategic feature engineering, involving the elimination of uncorrelated and multicollinearity features, as well as the creation of new data bins. Addressing missing values and standardizing value names were crucial steps, alongside normalizing numeric attributes. For modeling, seven classification models were constructed, trained, and rigorously evaluated. The pinnacle model underwent meticulous hyperparameter tuning, finalizing an efficient predictive framework.

**Findings and Learnings:** <br>
The findings reveal that the model demonstrates strong accuracy in detecting non-defaulters. However, its performance is notably weaker in identifying defaulters. This indicates the model's effectiveness in classifying non-defaulters while highlighting the need for improvement when recognizing defaulters to enhance overall predictive capabilities. Raw data proved challenging due to the extensive cleaning it had to go through before being used for modeling.

***

[Part 1](https://github.com/Deuellau/Projects/blob/main/Loan%20Approval/Loan%20Approval%20(Part%201).ipynb) covers Data Understanding and Exploratory Data Analysis. <br>
[Part 2](https://github.com/Deuellau/Projects/blob/main/Loan%20Approval/Loan%20Approval%20(Part%202).ipynb) covers Data Preparation, Modeling and Evaluation.

# Understanding Obesity (Prediction)

The objective of this project is to accurately classify individuals into different weight categories based on relevant features. We will explore the data and features, compute various machine learning algorithms with various parameters for optimisation, and evaluate their performance to identify the most effective model.

**Environment Setup**



```python
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
%matplotlib inline

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
```

## Data Understanding

### Dataset Information


```python
df = pd.read_csv('ObesityDataSet.csv')

nominal_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 
                   'SMOKE', 'SCC', 'MTRANS']
df[nominal_columns] = df[nominal_columns].astype('category')

ordinal_columns = ['CAEC', 'CALC', 'NObeyesdad']
category_order = {'CAEC': ['no', 'Sometimes', 'Frequently', 'Always'],
                  'CALC': ['no', 'Sometimes', 'Frequently', 'Always'],
                  'NObeyesdad': ['Insufficient_Weight', 'Normal_Weight', 
                                 'Overweight_Level_I', 'Overweight_Level_II', 
                                 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']}

for col in ordinal_columns:
    df[col] = pd.Categorical(df[col], categories=category_order[col], ordered=True)

    
display(df.head())
df.info()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>family_history_with_overweight</th>
      <th>FAVC</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CAEC</th>
      <th>SMOKE</th>
      <th>CH2O</th>
      <th>SCC</th>
      <th>FAF</th>
      <th>TUE</th>
      <th>CALC</th>
      <th>MTRANS</th>
      <th>NObeyesdad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>21.0</td>
      <td>1.62</td>
      <td>64.0</td>
      <td>yes</td>
      <td>no</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>no</td>
      <td>Public_Transportation</td>
      <td>Normal_Weight</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>21.0</td>
      <td>1.52</td>
      <td>56.0</td>
      <td>yes</td>
      <td>no</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>yes</td>
      <td>3.0</td>
      <td>yes</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Normal_Weight</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>23.0</td>
      <td>1.80</td>
      <td>77.0</td>
      <td>yes</td>
      <td>no</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>2.0</td>
      <td>no</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Frequently</td>
      <td>Public_Transportation</td>
      <td>Normal_Weight</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>27.0</td>
      <td>1.80</td>
      <td>87.0</td>
      <td>no</td>
      <td>no</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>2.0</td>
      <td>no</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>Frequently</td>
      <td>Walking</td>
      <td>Overweight_Level_I</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>22.0</td>
      <td>1.78</td>
      <td>89.8</td>
      <td>no</td>
      <td>no</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Overweight_Level_II</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2111 entries, 0 to 2110
    Data columns (total 17 columns):
     #   Column                          Non-Null Count  Dtype   
    ---  ------                          --------------  -----   
     0   Gender                          2111 non-null   category
     1   Age                             2111 non-null   float64 
     2   Height                          2111 non-null   float64 
     3   Weight                          2111 non-null   float64 
     4   family_history_with_overweight  2111 non-null   category
     5   FAVC                            2111 non-null   category
     6   FCVC                            2111 non-null   float64 
     7   NCP                             2111 non-null   float64 
     8   CAEC                            2111 non-null   category
     9   SMOKE                           2111 non-null   category
     10  CH2O                            2111 non-null   float64 
     11  SCC                             2111 non-null   category
     12  FAF                             2111 non-null   float64 
     13  TUE                             2111 non-null   float64 
     14  CALC                            2111 non-null   category
     15  MTRANS                          2111 non-null   category
     16  NObeyesdad                      2111 non-null   category
    dtypes: category(9), float64(8)
    memory usage: 152.2 KB


### Data Description

0. **Gender** - Gender
1. **Age** - Age
2. **Height** - Height
3. **Weight** - Weight
4. **family_history_with_overweight** - Family members suffered or suffers from overweight
5. **FAVC** - Frequent consumption of high caloric food
6. **FCVC** - Frequency of consumption of vegetables
7. **NCP** - Number of main meals
8. **CAEC** - Consumption of food between meals
9. **SMOKE** - Smoker or not
10. **CH20** - Consumption of water daily
11. **SCC** - Calories consumption monitoring
12. **FAF** - Physical activity frequency
13. **TUE** - Time using technology devices
14. **CALC** - Consumption of alcohol
15. **MTRANS** - Transportation used
16. **NObeyesdad** - Obesity level deducted


NObeyesdad values are computing using BMI values and categorised based on the following:
- Underweight - Less than 18.5
- Normal - 18.5 to 24.9
- Overweight - 25.0 to 29.9 (Unclear how Overweight I and II are divided)
- Obesity I - 30.0 to 34.9
- Obesity II - 35.0 to 39.9
- Obesity III - Higher than 40

### Missing Values


```python
df.isna().sum()
```




    Gender                            0
    Age                               0
    Height                            0
    Weight                            0
    family_history_with_overweight    0
    FAVC                              0
    FCVC                              0
    NCP                               0
    CAEC                              0
    SMOKE                             0
    CH2O                              0
    SCC                               0
    FAF                               0
    TUE                               0
    CALC                              0
    MTRANS                            0
    NObeyesdad                        0
    dtype: int64



### Descriptive Statistics


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CH2O</th>
      <th>FAF</th>
      <th>TUE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>24.312600</td>
      <td>1.701677</td>
      <td>86.586058</td>
      <td>2.419043</td>
      <td>2.685628</td>
      <td>2.008011</td>
      <td>1.010298</td>
      <td>0.657866</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.345968</td>
      <td>0.093305</td>
      <td>26.191172</td>
      <td>0.533927</td>
      <td>0.778039</td>
      <td>0.612953</td>
      <td>0.850592</td>
      <td>0.608927</td>
    </tr>
    <tr>
      <th>min</th>
      <td>14.000000</td>
      <td>1.450000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>19.947192</td>
      <td>1.630000</td>
      <td>65.473343</td>
      <td>2.000000</td>
      <td>2.658738</td>
      <td>1.584812</td>
      <td>0.124505</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>22.777890</td>
      <td>1.700499</td>
      <td>83.000000</td>
      <td>2.385502</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.625350</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>26.000000</td>
      <td>1.768464</td>
      <td>107.430682</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.477420</td>
      <td>1.666678</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>61.000000</td>
      <td>1.980000</td>
      <td>173.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe(include='category')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>family_history_with_overweight</th>
      <th>FAVC</th>
      <th>CAEC</th>
      <th>SMOKE</th>
      <th>SCC</th>
      <th>CALC</th>
      <th>MTRANS</th>
      <th>NObeyesdad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2111</td>
      <td>2111</td>
      <td>2111</td>
      <td>2111</td>
      <td>2111</td>
      <td>2111</td>
      <td>2111</td>
      <td>2111</td>
      <td>2111</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Male</td>
      <td>yes</td>
      <td>yes</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>no</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Obesity_Type_I</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1068</td>
      <td>1726</td>
      <td>1866</td>
      <td>1765</td>
      <td>2067</td>
      <td>2015</td>
      <td>1401</td>
      <td>1580</td>
      <td>351</td>
    </tr>
  </tbody>
</table>
</div>



### Key Findings

- 2,111 rows.
- 16 features.
- 0 missing values.
- Target variable: 'NObeyesdad'.




* Average age is relatively young at 24 years old.
* Average height and weight are 1.70m and 86.6kg.
* Most individuals have family history of obesity and consume high caloric food.
* Public transport being the most common type of transportation.

## Exploratory Data Analysis

### Target Variable


```python
plt.figure(figsize=(8,5))
sns.countplot(x='NObeyesdad', data=df, hue='NObeyesdad', palette='icefire')
plt.title('NObeyesdad')
plt.xticks(rotation=20)
plt.show()
```


    
![png](output_16_0.png)
    


- The number of individuals increases with each higher weight classification.
- Obesity Type 1 and Insufficient Weight are the two most and least occuring weight categories.
- Distribution of the categories are notably proportionate, which suggests a good balanced representatiton of each weight category.

### Demographic features

**Gender**


```python
plt.figure(figsize=(8,5))
palette_gender = {'Male': '#2986cc', 'Female': '#c90076'}
sns.countplot(x='NObeyesdad', data=df, hue='Gender', palette=palette_gender)
plt.title('Gender')
plt.xticks(rotation=20)
plt.show()
```


    
![png](output_20_0.png)
    


- The weight category most frequently observed among females proves to be the least common category among males, and vice versa.
- Higher number of males fall into the Overweight categories compared to females. In contrast, more females are in the Insufficient weight and Obesity categories compared to males.
- Almost twice as many females fall into the heaviest weight class compared to any other female weight class.

**Age**


```python
plt.figure(figsize=(8,5))
sns.boxplot(x='NObeyesdad', y='Age', data=df, hue='NObeyesdad', palette='icefire')
plt.title('Age')
plt.xticks(rotation=20)
plt.show()
```


    
![png](output_23_0.png)
    


- Individuals in heavier weight categories tend to be older, as they are more likely to fall into higher weight classifications as they age.
- Overweight Level II has the greatest variability, suggesting that this category may be diverse and not strictly tied to a specific age group, while Insufficient Weight tends occur within individuals who are less than 30 years old
- The Obesity Type III has a maximum age value of less than 30, while Normal Weight has an individual with age exceeding 60, suggesting that weight classes has a potential impact on life expectancy.

**Weight and Height**


```python
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x='Weight', y='Height', data=df, hue='NObeyesdad', palette='icefire')
plt.title('Weight vs Height - Healthy')

plt.subplot(1, 2, 2)
sns.scatterplot(x='Weight', y='Height', data=df, hue='Gender', palette=palette_gender)
plt.title('Weight vs Height - Gender')

plt.tight_layout()
plt.show()
```


    
![png](output_26_0.png)
    


- Individuals with heavier weights, particularly when not proportionately compensated with taller heights, tend to fall into the heavier weight categories. This suggests that a disparity between weight and height may contribute to the classification of individuals into higher weight categories.
- Females are also seen with lower weights and shorter heights compared to males, as expected.

**Family history with overweight**


```python
plt.figure(figsize=(8, 5))
palette_yesno = {'yes': '#3f8f29', 'no': '#bf1029'}
sns.countplot(x='NObeyesdad', data=df, hue='family_history_with_overweight', palette=palette_yesno)
plt.title('Family History')
plt.xticks(rotation=20)
plt.show()
```


    
![png](output_29_0.png)
    


- Individuals with no family history of being overweight are observed to have more individuals in the lighter weight classes. Likewise, more individuals are in the heavier categories when their family has a history of overweight.

### Eating Habit Features

**Consumption of high caloric food**


```python
plt.figure(figsize=(8,5))
sns.countplot(x='NObeyesdad', data=df, hue='FAVC', palette=palette_yesno)
plt.title('Frequency of consumption of high caloric food')
plt.xticks(rotation=20)
plt.show()
```


    
![png](output_33_0.png)
    


- More individuals in heavier weight classes are seen to frequently consume high caloric food than those in lighter weight classes.
- Likewise, those in lighter weight classes do not frequently consume high caloric food.

**Consumption of vegetables**


```python
plt.figure(figsize=(8,5))
sns.violinplot(x='NObeyesdad', y='FCVC', data=df, hue='NObeyesdad', palette='icefire')
plt.title('Frequency of consumption of vegetables')
plt.xticks(rotation=20)
plt.ylim(df['FCVC'].min()-0.1, df['FCVC'].max()+0.1)
plt.show()
```


    
![png](output_36_0.png)
    


- Frequency of consuming vegetables decrease as with an increase of weight classes , specfically from a majority of individuals consuming more than 2.5 to a larger proportion opting for 2.0, except for Obesity Type II where almost an equal number of individuals from 2.0 to 3.0
- Obesity Type III has all indivdiduals with a frequency of 3.0

**Number of main meals**


```python
plt.figure(figsize=(8,5))
sns.violinplot(x='NObeyesdad', y='NCP', data=df, hue='NObeyesdad', palette='icefire')
plt.title('Number of main meals')
plt.xticks(rotation=20)
plt.ylim(df['NCP'].min()-0.1, df['NCP'].max()+0.1)
plt.show()
```


    
![png](output_39_0.png)
    


- All of the weight classes generally consume three meals. 
- However, ironically, the Insufficient Weight class has more individuals consuming more than 3 meals than other weight classes.

**Consumption of food between meals**


```python
plt.figure(figsize=(8,5))
sns.countplot(x='NObeyesdad', data=df, hue='CAEC', palette='magma')
plt.title('Consumption of food between meals')
plt.xticks(rotation=20)
plt.show()
```


    
![png](output_42_0.png)
    


- Individuals generally consume food between their meals.
- Lighter weight classes are seen to consume more food inbetween their meals compared to heavier weight classes.

**Consumption of water daily**


```python
plt.figure(figsize=(8,5))
sns.boxplot(x='NObeyesdad', y='CH2O', data=df, hue='NObeyesdad', palette='icefire')
plt.title('Consumption of water daily')
plt.xticks(rotation=20)
plt.ylim(df['CH2O'].min(), df['CH2O'].max())
plt.show()
```


    
![png](output_45_0.png)
    


- Individuals generally consume the same amount of water.
- Water consumption increases as the weight classes get heavier.

**Consumption of alcohol**


```python
plt.figure(figsize=(8,5))
sns.countplot(x='NObeyesdad', data=df, hue='CALC', palette='magma')
plt.title('Consumption of alcohol')
plt.xticks(rotation=20)
plt.show()
```


    
![png](output_48_0.png)
    


- Alcohol consumption generally increases as the weight classes get heavier.
- Obesity Type III has almost no individuals not consuming alcohol, while the middle weight classes display a diverse range of alcohol consumption habits.

### Physical Habit Features 

**Smoking**


```python
plt.figure(figsize=(8,5))
sns.countplot(x='NObeyesdad', data=df, hue='SMOKE', palette=palette_yesno)
plt.title('Smoking')
plt.xticks(rotation=20)
plt.show()
```


    
![png](output_52_0.png)
    


- Majority of individuals in each class do not smoke, while a small group in the middle weight classes smoke.

**Calories consumption monitoring**


```python
plt.figure(figsize=(8,5))
sns.countplot(x='NObeyesdad', data=df, hue='SCC', palette=palette_yesno)
plt.title('Calories consumption monitoring')
plt.xticks(rotation=20)
plt.show()
```


    
![png](output_55_0.png)
    


- Individuals across the weight categories are seen to not monitor their calories intake.
- It is only seen in the lighter weight categories that a small group of individuals monitor their intake.

**Physical activity frequency**


```python
plt.figure(figsize=(8,5))
sns.boxplot(x='NObeyesdad', y='FAF', data=df, hue='NObeyesdad', palette='icefire')
plt.title('Physical activity frequency')
plt.xticks(rotation=20)
plt.ylim(df['FAF'].min(), df['FAF'].max())
plt.show()
```


    
![png](output_58_0.png)
    


- The frequency of physical activity decreases slightly as the weight classes get heavier.
- Only Obesity Type II and III have no individuals with a frequency of 2.0 and above, suggesting that these two classes generally have lower physical activity frequency compared to the other weight classes.

**Time using technology devices**


```python
plt.figure(figsize=(8,5))
sns.kdeplot(x='TUE', data=df, hue='NObeyesdad', palette='icefire')
plt.title('Time using technology devices')
plt.xticks(rotation=20)
plt.xlim(df['TUE'].min(), df['TUE'].max())
plt.show()
```


    
![png](output_61_0.png)
    


- Individuals tend to have a greater number of people who spend less time on devices.
- Obesity Type III is also seen to have a greater number of individuals spending more time on devices.

**Transportation used**


```python
plt.figure(figsize=(8,5))
sns.countplot(x='NObeyesdad', data=df, hue='MTRANS', palette='magma')
plt.title('Consumption of food between meals')
plt.xticks(rotation=20)
plt.show()
```


    
![png](output_64_0.png)
    


- Public Transport is the most common mode of transportation, followed by automobile.
- Middle weight classes have more individuals using the automobile, while also having lesser individuals using the public transport.
- Normal Weight is also seen to have the most diversity of transportation used.

### Correlation Analysis


```python
continuous_columns = df.select_dtypes(include='float64')
corr = continuous_columns.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10,5))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=.5, cmap='magma')
plt.show()
```


    
![png](output_67_0.png)
    


- As age increases, time spend on devices tend to decrease slightly.
- As height increases, weight tends to increase moderately.
- Taller individuals tend to slightly have more meals, drink more water daily and have more frequent physical activities.
- As weight increases, consumption of vegetables and daily water consumption increases slightly.

### Key Findings

Individuals in the heavier weight classes tend to:
- be older.
- be shorter or heavier than usual
- have a family history of overweight
- consume high caloric food
- have a higher frequency of vegetable consumption
- consume lesser or do not consume food between meals
- have slightly higher daily water consumption
- sometimes consume alcohol
- not monitor their calories consumption
- have lower frequency of physical activity
- spend more time on technology devices
- take the public transport and have a low diversity when selecting a transportation to use

## Data Preparation

**Feature Selection**

We will exclude the Height and Weight features, as the target variable computed using Height and Weight and categorised into the different weight classes.


```python
df = df.drop(['Height', 'Weight'], axis=1)
```

**Outlier Detection and Removal**

We will remove outliers as they are data points that deviate significantly from the majority of the dataset and can introduce noise and distort the learning process of machine learning model. We will iterate through each feature and identify if its quantitative, calculate its IQR, lower and upper bounds and and retain values falling within these bounds.

For this case, the multiplier choice would be 3, hence any value below Q1 - 3 * IQR or above Q3 + 3 * IQR is considered an outlier.


```python
for column in df:
    if df[column].dtype == 'float64':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        df = df[
            (df[column] >= lower) & (df[column] <= upper)]
        
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 1762 entries, 0 to 2110
    Data columns (total 15 columns):
     #   Column                          Non-Null Count  Dtype   
    ---  ------                          --------------  -----   
     0   Gender                          1762 non-null   category
     1   Age                             1762 non-null   float64 
     2   family_history_with_overweight  1762 non-null   category
     3   FAVC                            1762 non-null   category
     4   FCVC                            1762 non-null   float64 
     5   NCP                             1762 non-null   float64 
     6   CAEC                            1762 non-null   category
     7   SMOKE                           1762 non-null   category
     8   CH2O                            1762 non-null   float64 
     9   SCC                             1762 non-null   category
     10  FAF                             1762 non-null   float64 
     11  TUE                             1762 non-null   float64 
     12  CALC                            1762 non-null   category
     13  MTRANS                          1762 non-null   category
     14  NObeyesdad                      1762 non-null   category
    dtypes: category(9), float64(6)
    memory usage: 113.4 KB


**Feature Engineering**

To enhance model performance, we will convert categorical features that represent qualitative data into a numerical format for models to interpret them effectively. 

We will achieve this by one-hot encoding nominal features and label encoding ordinal features.


```python
df = pd.get_dummies(df, columns=nominal_columns, drop_first=True)

label_encoder = LabelEncoder()
for column in ordinal_columns:
    df[column] = label_encoder.fit_transform(df[column])
```

**Data Normalisation**

Data normalisation standardises the scale of features, making them comparable and preventing features with larger values from dominating features with smaller values. We will also implement feature-wise normalisation, where the characteristic of the feature dictates the type of normalisation technique used.

Min-Max scaling will be used for features with specific range of values, while Z-score for features with distributions that are approximately normal.


```python
scaler = StandardScaler()

minmax_columns = ['FCVC', 'NCP', 'CH2O']
zscore_columns = ['Age']

minmax_scaler = MinMaxScaler()
df[minmax_columns] = minmax_scaler.fit_transform(df[minmax_columns])

zscore_scaler = StandardScaler()
df[zscore_columns] = zscore_scaler.fit_transform(df[zscore_columns])

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CAEC</th>
      <th>CH2O</th>
      <th>FAF</th>
      <th>TUE</th>
      <th>CALC</th>
      <th>NObeyesdad</th>
      <th>Gender_Male</th>
      <th>family_history_with_overweight_yes</th>
      <th>FAVC_yes</th>
      <th>SMOKE_yes</th>
      <th>SCC_yes</th>
      <th>MTRANS_Bike</th>
      <th>MTRANS_Motorbike</th>
      <th>MTRANS_Public_Transportation</th>
      <th>MTRANS_Walking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.541299</td>
      <td>0.5</td>
      <td>0.572509</td>
      <td>2</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.541299</td>
      <td>1.0</td>
      <td>0.572509</td>
      <td>2</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.198114</td>
      <td>0.5</td>
      <td>0.572509</td>
      <td>2</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.488255</td>
      <td>1.0</td>
      <td>0.572509</td>
      <td>2</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>5</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.831440</td>
      <td>0.5</td>
      <td>0.572509</td>
      <td>2</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Summary of Data Preparation

- Identified and removed outliers from quantitative features.
- One-hot encoded categorical features.
- Normalised quantitative features.

## Modeling

**Train-test Split**

We will divide the dataframe into training and testing sets. Our testing set will include 30% of the dataset that is stratified on our target variable to maintain the distribution of the target variable in both training and testing sets.


```python
X = df.drop(['NObeyesdad'], axis=1)
y = df['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```

**Model Training and Evaluation Function**

We will create a function to simplify the process of training and evaluating an algorithm by automating hyperparameter tuning. The function performs a grid search to optimise model performance and updates the best performance into a dataframe. The function also reports the elapsed time for the entire search and outputs the best parameters found during the search.


```python
metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
f1_scorer = make_scorer(f1_score, average='weighted')
skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def train_and_evaluate(name, model, model_params):
    print('Algorithm:',name)
    start_time = time.time()
    
    
    grid_search = GridSearchCV(model, model_params, scoring=f1_scorer, 
                               cv=skfold.split(X_train, y_train), verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f'Elapsed time: {elapsed_time:.2f}s')
    print('Best parameters:', grid_search.best_params_)
    print('')
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(y_test, y_pred, average='weighted') * 100, 2)
    recall = round(recall_score(y_test, y_pred, average='weighted') * 100, 2)
    f1 = round(f1_score(y_test, y_pred, average='weighted') * 100, 2)
    
    metrics_df.loc[len(metrics_df)] = [name, accuracy, precision, recall, f1]
```

**Hyperparameter Tuning and Model Evaluation**

We will define the hyperparameter configuration for various classification algorithms. These configurations, consisting of algorithms and their respective parameters, will be used as input for the previously defined function. The parameter choices aim to find a balance between efficient run time and the number of fits during the model tuning process.


```python
lgr_params = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 0.5, 1, 2],
    'max_iter': [3000, 4000, 5000, 6000],
    'solver': ['saga' ],
    'multi_class': ['ovr', 'multinomial'],
    'class_weight': [None, 'balanced'],
    'warm_start': [True, False],
    'random_state': [42]
}

svm_params = {
    'C': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [1, 2, 3, 4, 5],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'coef0': [0, 1, 2, 3],
    'random_state': [42]
}

dst_params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 2, 3, 4, 5, 8, 10, 15, 20],
    'min_samples_split': [2, 3, 4, 5, 8, 10, 15],
    'min_samples_leaf': [1, 2, 3, 4, 5, 8, 10, 20],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3],
    'random_state': [42]
}

rfr_params = {
    'n_estimators': [300, 400, 500, 600],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 2, 3, 5, 7],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced'],
    'random_state': [42]
}

gbo_params = {
    'learning_rate': [0.05, 0.1, 0.5],
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 2, 3, 5, 7, 9],
    'random_state': [42]
}

xgb_params = {
    'learning_rate': [0.05, 0.1, 0.5],
    'n_estimators': [100, 300, 500, 700],
    'max_depth': [None, 2, 3, 5, 7, 9],
    'random_state': [42]
}

models = {
    'Logistic Regression': (LogisticRegression(), lgr_params),
    'Support Vector Machine': (SVC(), svm_params),
    'Decision Tree': (DecisionTreeClassifier(), dst_params),
    'Random Forest': (RandomForestClassifier(), rfr_params),
    'Gradient Boosting': (GradientBoostingClassifier(), gbo_params),
    'Extreme Gradient Boosting': (XGBClassifier(), xgb_params)
}

for name, (model, params) in models.items():
    train_and_evaluate(name, model, params)
```

    Algorithm: Logistic Regression
    Fitting 10 folds for each of 256 candidates, totalling 2560 fits
    Elapsed time: 48.94s
    Best parameters: {'C': 1, 'class_weight': 'balanced', 'max_iter': 3000, 'multi_class': 'multinomial', 'penalty': 'l2', 'random_state': 42, 'solver': 'saga', 'warm_start': True}
    
    Algorithm: Support Vector Machine
    Fitting 10 folds for each of 2880 candidates, totalling 28800 fits
    Elapsed time: 108.78s
    Best parameters: {'C': 8, 'coef0': 0, 'degree': 5, 'gamma': 0.1, 'kernel': 'poly', 'random_state': 42}
    
    Algorithm: Decision Tree
    Fitting 10 folds for each of 12096 candidates, totalling 120960 fits
    Elapsed time: 39.21s
    Best parameters: {'criterion': 'gini', 'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 42, 'splitter': 'random'}
    
    Algorithm: Random Forest
    Fitting 10 folds for each of 240 candidates, totalling 2400 fits
    Elapsed time: 113.54s
    Best parameters: {'bootstrap': False, 'class_weight': None, 'criterion': 'entropy', 'max_depth': None, 'n_estimators': 500, 'random_state': 42}
    
    Algorithm: Gradient Boosting
    Fitting 10 folds for each of 54 candidates, totalling 540 fits
    Elapsed time: 170.48s
    Best parameters: {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 300, 'random_state': 42}
    
    Algorithm: Extreme Gradient Boosting
    Fitting 10 folds for each of 72 candidates, totalling 720 fits
    Elapsed time: 124.26s
    Best parameters: {'learning_rate': 0.05, 'max_depth': 9, 'n_estimators': 700, 'random_state': 42}
    


# Results and Conclusion

### Model Evaluation

**Model Performance**


```python
metrics_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>58.03</td>
      <td>56.24</td>
      <td>58.03</td>
      <td>56.40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Support Vector Machine</td>
      <td>78.45</td>
      <td>78.07</td>
      <td>78.45</td>
      <td>78.06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Decision Tree</td>
      <td>73.16</td>
      <td>72.99</td>
      <td>73.16</td>
      <td>73.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Random Forest</td>
      <td>86.20</td>
      <td>86.79</td>
      <td>86.20</td>
      <td>86.21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gradient Boosting</td>
      <td>84.88</td>
      <td>85.61</td>
      <td>84.88</td>
      <td>84.92</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Extreme Gradient Boosting</td>
      <td>83.74</td>
      <td>84.16</td>
      <td>83.74</td>
      <td>83.76</td>
    </tr>
  </tbody>
</table>
</div>



We can see that most models performed resonably well except for Logistic Regression. Overall, the ensemble methods, such as Random Forest and Gradient Boosting, performed better than individual models like Logistic Regression and Decision Tree.

Since Random Forest performed was the best performer across all metrics, we will create a new model using it along with the identified optimal parameters. This helps us better understand its classification capabilities.


```python
rfr_best_params = {'bootstrap': False, 'class_weight': None, 'criterion': 'entropy', 'max_depth': None, 'n_estimators': 500, 'random_state': 42}
rfr_model = RandomForestClassifier(**rfr_best_params)
rfr_model.fit(X_train, y_train)
y_pred = rfr_model.predict(X_test)
```

**Feature Importance**

Lets delve into the optimal model to gain insights into the key features it relies on for effective prediction of the target variable. This analysis will help us identify the most important features, providing valuable information about the factors contributing significantly to the model's predictive capabilities.


```python
feature_names = X.columns
feature_importances = rfr_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

sns.set_palette('icefire')
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()
```


    
![png](output_104_0.png)
    



```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

top5 = len(feature_importance_df) - 5
axes[0].barh(feature_importance_df['Feature'][top5:], feature_importance_df['Importance'][top5:], color='#3f8f29')
axes[0].set_xlabel('Feature Importance')
axes[0].set_title('Top 5 Features')

btm5 = 5
axes[1].barh(feature_importance_df['Feature'][:btm5], feature_importance_df['Importance'][:btm5], color='#bf1029')
axes[1].set_xlabel('Feature Importance')
axes[1].set_title('Bottom 5 Features')

plt.tight_layout()
plt.show()
```


    
![png](output_105_0.png)
    


We can see that vegetable consumption frequency, age, gender, time using technology devices and physical activity frequency has the biggest impact in accurate classification.

In contrast, monitoring calories consumption, type of transport used and smoking has the least impact in accurate classification.

**Confusion Matrix**


```python
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=rfr_model.classes_, yticklabels=rfr_model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```


    
![png](output_108_0.png)
    


**Classification Report**


```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.91      0.90      0.91        69
               1       0.66      0.81      0.73        70
               2       0.86      0.78      0.82        81
               3       0.92      0.98      0.95        81
               4       0.98      1.00      0.99        97
               5       0.79      0.79      0.79        63
               6       0.89      0.71      0.79        68
    
        accuracy                           0.86       529
       macro avg       0.86      0.85      0.85       529
    weighted avg       0.87      0.86      0.86       529
    


The model achieving an overall accuracy of 86% indicates a strong ability to correctly classify the target variable. The precision and recall values across the different weight classes are generally high, indicating the model's ability to identify specific classes.

It's worth highlightly that the model was able to accurately identifying all individuals with Obesity Type I (class 4). This may come from the fact that this class has the largest number of instances in the dataset. The model performs the worst at classifying Normal Weight (class 1) may suggest that the data used might not provide sufficient discriminative features for accurately classifying individuals with Normal Weight, unlike the other classes.

### Conclusion

In conclusion, the model evaluation reveals various performance among different algorithms, with ensemble methods like Random Forest and Gradient Boosting outperforming individual models such as Logistic Regression and Decison Tree. Random Forest together with its optimal parameters resulted in the best overall performance across accuracy, precision, recall, and F1 score metrics.

The feature importance analysis highlights key factors influencing the model's predictions, with vegetable consumption frequency, age, gender, time using technology devices, and physical activity frequency being the most impactful. Conversely, monitoring calorie consumption, type of transport used, and smoking have a relatively lower impact on accurate classification.

The model achieves an impressive accuracy of 86%, showcasing its ability to correctly classify individuals into weight categories. Notably, the model excels in identifying individuals with Obesity Type I, benefiting from the larger number of instances in this class. However, it faces challenges in accurately classifying individuals with Normal Weight, suggesting potential limitations in the available data for this particular category.

In summary, the model demonstrates strong predictive capabilities, especially in categories with ample data representation, while recognizing the need for further refinement, particularly for less represented classes.

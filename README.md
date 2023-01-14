---
### Case Study 2: Which Factors Effect a companies Profit ? 
#### Multiple Linear Regression Model
##### Nathaniel Nete-Sie Williams Jr.
##### 01/09/2022
---

### Importing Libraries 

```

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
%matplotlib inline
```
### Exploration

Now we look through the dataset 
```
companies = pd.read_csv("1000_Companies.csv")
companies
```
Now we're gonna make a correlation heat map to see how each variable correlates with our Y variable 'Profit'. 1 is the closest correlation and anything less than that is a weaker correlation 

I put color map 'cmap' color scheme that works for me and I added annotation 'annot=TRUE' in order to see the data on each cube 
```
sns.heatmap(companies.corr(),cmap="YlGnBu",annot=True)
```
![](https://github.com/netesie/Predicting-a-Companies-Profits-w-Linear-Regression-/blob/main/heatmap.png "Correlation Heatmap")


If you look at where 'R&D Spend' and 'Marketing Spend' meets you'll notice a value of .98. Any value over .90 between 2 indepenent values is NOT a good thing. It's called _multicollinearity_ and can ruin the accuracy of a model and introduce bias. In order to account for this we can add those 2 variables together or get rid of one. We'll get rid of them.

```
df = companies
del df['R&D Spend']
```

Now we are going to pre process our data by encoding our nominal categorical data in our 'state' column

### Encoding Categorical Variables

We assign the indepent variable columns into X. 

Then assign our dependent variable column 'Profit' into Y

Then check for nulls

```
#encoding categorical field 'State'
df['State'] = df['State'].astype('category')
df['State'] = df['State'].cat.codes

#0 is Cali, 1 is Florida, and 2 is New York

# let's check for null values
df.isnull().sum


#Split the Dataset 
X = df.drop(columns = 'Profit')

y = df['Profit']

X

```

Now we're going to delete the 'R&D Spend' column and the label headers from each column so its ready to be processed in the linear regression model

### Splitting the data into Train and Test set:

```
# We're going to assign the Test set size to 0.2 or 20% of the rows. 
# The Remaining 80% will be the Training Set
# The random state will randomly take 200 out of the 1000 rows of data to pull off to the side to test 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0 )
```
### Fitting Model to Training Set:

```
# This is how we train our model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Now we find the Y intercept
c = lr.intercept_
print (c)

#Then we find the regression coefficients for each factor/feature/independent variable
m = lr.coef_
print (m)

```
Now we know the training is done after checking the Y intercept value and the regression coefficients 

Next we try to predict the 'Profit' with our model using the training set

```
y_pred_train = lr.predict(X_train)
y_pred_train


#Now check how well the model did its prediction by setting up a scatter plot and look at the correlation.
plt.scatter(y_train, y_pred_train)
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
```
![](https://github.com/netesie/Predicting-a-Companies-Profits-w-Linear-Regression-/blob/main/model%20v%20train%20set.png " Model vs Train Set ")
```
# R2 score of the training set
r2_score(y_train, y_pred_train)
```
0.9458030630681264

### Testing the Model

```
# Now we use the testing data to test the model and see its ability to generalize
y_pred_test = lr.predict(X_test)

#Now check again to see how well the model predicts by ...
#...setting up a scatter plot and looking at the correlation.
plt.scatter(y_test, y_pred_test)
plt.xlabel('Actual Profit')   
plt.ylabel('Predicted Profit')
```

![](https://github.com/netesie/Predicting-a-Companies-Profits-w-Linear-Regression-/blob/main/model%20v%20test%20set.png " Model vs Test Set ")

### R Squared  and Adjusted R Squared of Test Set 

```
# Calculating the R squared value to see if its around .80 or above to see that this is a valid model
# This means that 89% of the variation in the response (dependent) variable can be explained... 
# ...by the predictor variables in the model.

r2_score(y_test, y_pred_test)
```
0.8985038788872521

```
# Calculate adjusted R2 of the train set so multiple predictors dont skew the R2
# Adj r2 = 1-(1-R2)*(n-1)/(n-p-1)

Adj_r2 = 1-(1-0.8985038788872521)*(200-1)/(200-2-1)

Adj_r2
```
0.8974734614140263

### Model Validation (Residual Analysis) 

```
residuals = y_test-y_pred_test

sns.regplot(x = y_pred_test, y = residuals, data = None, scatter = True, color = 'red')
```
![](https://github.com/netesie/Predicting-a-Companies-Profits-w-Linear-Regression-/blob/main/Residual%20Plot.png " Residual Plot ")

As we can see our Residual Plot doesnt vary too much at all and are all clusteered around one line 

### Conclusion

Profit at this company is closely correlated to putting money into R&D and Market spend. Now we can safely predict profits based on how much money we put into each department.





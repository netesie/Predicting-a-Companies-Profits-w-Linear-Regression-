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

If you look at where 'R&D Spend' and 'Marketing Spend' meets you'll notice a value of .98. Any value over .90 between 2 indepenent values is NOT a good thing. It's called multicollinearity and can ruin the acuracy of a model and introduce bias. In order to account for this we can add those 2 variables together or get rid of one. We'll get rid of them.

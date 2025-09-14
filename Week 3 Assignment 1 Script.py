from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


iris = datasets.load_iris()
weight_data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 
                           5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 
                           6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 
                           5.37, 5.29, 4.92, 6.15, 5.80, 5.26], 
                           "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}

PlantGrowth = pd.DataFrame(weight_data)

X = iris.data  
y = iris.target

df = pd.DataFrame(data=X, columns=iris.feature_names)

df['species'] = iris.target_names[iris.target]

# Make a histogram of the variable Sepal.Width.

sw = df['sepal width (cm)']

# plt.hist(sw, edgecolor='white')
# plt.show()

#Based on the histogram from #1a, which would you expect to be higher, 
#the mean or the median? Why?

#answer: I would say they are really close, with median being just slightly larger

# print(sw.mean())
# print(sw.median())

#answer: 
# 3.057337
# 3.0

# Only 27% of the flowers have a Sepal.Width higher than ________ cm.

twenty_seven = sw.quantile(0.73)

# print(f"Only 27% of the flowers have a sepal width higher than {twenty_seven} cm")

print(df)

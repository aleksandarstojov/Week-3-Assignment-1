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

rows, cols = df.shape

print(df)

#print(rows, cols)

# Create a new data frame named sub1 which includes only the first 9 rows and the last row of iris. 
# Display sub1

first_nine_rows = df.head(9)

last_row = df.iloc[[-1]]

sub1 = pd.concat([first_nine_rows, last_row])

# print(sub1)



# Create a new data frame named sub2 which includes only rows in iris where the Sepal.
# Width is less than 2.4 and includes only the columns Sepal.Length, 
# Sepal.Width, and Species/target. Display sub2.

sub2 = df.loc[df['sepal width (cm)'] < 2.4,
              ['sepal length (cm)', 'sepal width (cm)', 'species']]

# print(sub2)

# Create a vector named Versicolor_Is_The_Best, which has the value 100 whenever Species/target is 
# equal to "versicolor" and has the value 0 otherwise. 
# Display Versicolor_Is_The_Best.

Versicolor_Is_The_Best = []

for sp in df['species']:
    if sp == 'versicolor':
        Versicolor_Is_The_Best.append(100)
    else:
        Versicolor_Is_The_Best.append(0)

Versicolor_Is_The_Best = pd.Series(Versicolor_Is_The_Best)

print(Versicolor_Is_The_Best)


#Save the column named Sepal.Width as its own vector named sw. 
# Use functions to find the mean, median, maximum, and minimum of sw.

sw = df['sepal width (cm)']

min = min(sw)
max = max(sw)
sw_mean = np.mean(sw)
sw_median = np.median(sw)

# print(f"Min: {min}, max: {max}, mean: {sw_mean}, median: {sw_median}")


#Use a loop to add up the values in sw one at a time until the sum first exceeds 100. 
# What is this sum, and how many times did the loop have to execute to reach it?
# For example, if we were adding up the values in sw until the sum first exceeded 10 instead, 
# the answer would be 3.5+3.0+3.2+3.1 = 12.8 after 4 loops.
# Also, be aware this could be done more efficiently in R without using loops, but I want you to get the practice.

sum = 0.0
counter = 0
 

for value in sw:
    sum += value  
    counter += 1      
    if sum > 100:  
        break        

print("The sum when is:", sum)
print("The number of loops is:", counter)

# Create a new function called cmtoin() that converts centimeters to inches (1 inch = 2.54 cm). 
# The values in sw are currently recorded in centimeters. 
# Apply your function to sw and save the result as a new vector named sw_in. Display the first 7 values of sw_in.

# def cmtoin():
#     sw_in = []
#     for value in sw:               
#         sw_in.append(value / 2.54) 
#     print(sw_in[:7])

# cmtoin()

# Create a plot that compares Sepal.Length to Petal.Length. 
# Add some informative/interesting axis labels, colors, etc. 
# Have some fun with it.

# plt.scatter(df[df['species']=='setosa']['sepal length (cm)'],
#             df[df['species']=='setosa']['petal length (cm)'],
#             color='pink', label='Setosa', s=60, alpha=0.7)

# plt.scatter(df[df['species']=='versicolor']['sepal length (cm)'],
#             df[df['species']=='versicolor']['petal length (cm)'],
#             color='purple', label='Versicolor', s=60, alpha=0.7)

# plt.scatter(df[df['species']=='virginica']['sepal length (cm)'],
#             df[df['species']=='virginica']['petal length (cm)'],
#             color='blue', label='Virginica', s=60, alpha=0.7)

# sep_len = df['sepal length (cm)']
# pet_len = df['petal length (cm)']

# plt.xlabel("Sepal Length")
# plt.ylabel("Petal Length")
# plt.title("Comparing sepal length to petal length")
# plt.legend(title="Species")

# plt.show()

# Make a histogram of the variable Sepal.Width.
# plt.hist(sw)
# plt.show()


twenty_seven = round(sw.shape[0] * 27 / 100)

print((sw.sort_values(ascending=False).head(twenty_seven)).min())
print(sw.quantile(0.73))

# print(sw.sort_values(ascending=True))


# Based on the histogram from 1a, which would you expect to be higher, the mean or the median? Why?
# print(f"Mean is: {sw.mean()}") #3.0573333333333337
# print(f"Median is: {sw.median()}") #3.0

#Confirm your answer to 1b by actually finding these values.

#Only 27% of the flowers have a Sepal.Width higher than ________ cm.

sns.pairplot(df)
plt.show()

#Make scatterplots of each pair of the numerical variables in iris (There should be 6 pairs/plots).

#Based on 1e, which two variables appear to have the strongest relationship? 
# And which two appear to have the weakest relationship?


print(df)
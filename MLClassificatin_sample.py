pip install plotly --upgrade                                                    
pip -q install yellowbrick                                                      
                                                                               
import pandas as pd                                                             
import numpy as np                                                              
import seaborn as sns                                                           
import matplotlib.pyplot as plt                                                 
import plotly.express as px                                                     

# importing the database                                                        

database = pd.read_csv('database')                                              


## Visualizing the database                                                     

database                                                                        

database.head(10)                                                               
database.tails(10)                                                              

database.describe() # Show min/max                                              

database[database['category'] >= highvalue] ## Visualizing the outliers         

np.unique(database['category'], return_counts=True)                             

sns.countplot(x = database['category']);                                        

plt.hist(x = database['category']);                                             

graph = px.scatter_matrix(database, dimensions=['category', 'category2', 'category3'], color = 'default')
graph.show()                                                                    

## Treating outliers                                                            

database.loc[database['category'] < 0] ## Finding the outliers in the database  

database[database['category'] < 0] ## Alt way of finding outliers               

database2 = database.drop('category', axis = 1) ## Removing an entire column  (not recommended)
database2

database.index                                                                  

database3 = database.drop(database[database['category'] < 0].index) ## Removing the ou

database3.loc[database['category'] < 0] ## Double-checking for outliers         

## Getting the mean of the Values                                               

database.mean()                                                                 

database['category'].mean()                                                     

database['category'][database['category'] > 0].mean() ## Replacing outliers with the m

database.loc[database['category'] < 0, 'category'] = mean                       

## Checking for null values                                                     

database.isnull()                                                               

database.isnull().sum()                                                         

database.loc[pd.isnull(database['category'])]                                   

database['category'].fillna(database['category'].mean(), inplace = True) ## Replacing 

database.loc[pd.isnull(database['category'])]                                   

database.loc[(database['category'] == value)                                    

database.loc[database['category'].isin([value1, value2, value3])]               

type(database)                                                                  

## Creating variables                                                           

x_database= database.iloc[:, value1:value2].values                              

x_database                                                                      

type(x_database)       

y_database = database.iloc[:, targetvalue].values ## Setting the target value   

y_database                                                                      

type(y_database)                                                                

x_database[:,'category1'].min(), x_database[:,'category2'].min(), x_database[:,'category3'].min()
x_database[:,'category1'].max(), x_database[:,'category2'].max(), x_database[:,'category3'].max()

from sklearn.preprocessing import StandardScaler                                

scaler_db = StandardScaler()                                                    

x_database = scaler_db.fit_transform(x_database)                                

x_database[:,value1].min(), x_database[:,value2].min(), x_database[:,value3].min()
x_database[:,value1].max(), x_database[:,value2].max(), x_database[:,value3].max()

x_database                                                                      

## Training and test database                                                   

from sklearn.model_selection import train_test_split                            

x_database_train, x_database_test, y_database_train, y_database_test = train_test_spli

x_database_train.shape                                                          
y_database_train.shape                                                          
x_database_test.shape, y_database_test.shape                                    

# Saving Variables                                                              

import pickle                                                                   

with open('database.pkl', mode = 'wb') as f:                                    
pickle.dump([x_database_train, y_database_train, x_database_test, y_database_teste],





## Machine Learning Classification sample code
## Naive Bayes sample

from sklearn.naive_bayes import GaussianNB

import pickle
with open('path/to/database.pk1', 'rb') as f:
x_database_train, y_database_train, x_database_test, y_database_teste = pickle.loa

x_database_train.shape, y_database_train.shape ## Visualizing

x_database_test.shape, y_database_test.shape

naive_function = GaussianNB() ## Renaming function

naive_function.fit(x_database_train, y_database_train) ## Applying the algorithm

predictions = naive_function.predict(x_database_test)
predictions

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report ##

accuracy_score(y_database_test, predictions)

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(naive_function)
cm.fit(x_database_train, y_database_train)
cm.score(x_database_test, y_database_test)

print(classification_report(y_database_teste, predictions))






## Trees sample
from sklearn.tree import DecisionTreeClassifier

import pickle
with open('path/to/database.pk1', 'rb') as f:
x_database_train, y_database_train, x_database_test, y_database_teste = pickle.load(

x_database_train.shape, y_database_train.shape ## Visualizing

x_database_test.shape, y_database_test.shape

tree_function = RandomForestClassifier(n_estimators=40, criterion='entropy', random_st
tree_function.fit(x_database_train.shape, y_database_train.shape)

predictions = tree_function.predict(x_database_test)

from sklearn.metrics import accuracy_score, classification_report ## Evaluating the al
accuracy_score(y_database_test, predictions)

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(tree_function))
cm.fit(x_database_train, y_database_train)
cm.score(x_database_test, y_database_test)

print(classification_report(y_database_teste, predictions))

## Visualization of trees

from sklearn import tree
previsores = ['category', 'category2', 'category3']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (20,20))
tree.plot_tree(tree_database, feature_names=previsores, class_names=['0','1'], filled=
fig.savefig('image.png')






## Random Trees sample
from sklearn.ensemble import RandomForestClassifier

import pickle
with open('path/to/database.pk1', 'rb') as f:
x_database_train, y_database_train, x_database_test, y_database_teste = pickle.load(

x_database_train.shape, y_database_train.shape ## Visualizing
x_database_test.shape, y_database_test.shape

random_tree_function = DecisionTreeClassifier(criterion='entropy', random_state = 0)
random_tree_function.fit(x_ database_train.shape, y_database_train.shape)

predictions = random_tree_function.predict(x_predict_teste)
predictions

from sklearn.metrics import accuracy_score, classification_report ## Evaluating the al
accuracy_score(y_database_test, predictions)

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(random_tree_function))
cm.fit(x_database_train, y_database_train)
cm.score(x_database_test, y_database_test)







## Orange rules sample

pip install Orange3
import Orange

database = Orange.data.Table('/path/to/database') ## Importing database 

database.domain ## Visualizing the values

divided_database = Orange.evaluation.testing.sample(database, n = 0.25) ## Dividing te

divided_database ## Visualizing values
divided_database[0]
divided_database[1]
database_train = divided_database[1]
database_test = divided_database[0]
len(database_train), len(database_test)

cn2 = Orange.classification.rules.CN2Learner() ## Renaming function
rules_database = cn2(database_train)

for rules in rules_database.rule_list: ## Visualizing the rules made
print(rules)

predictions = Orange.evaluation.testing.TestOnTestData(database_train, database_train,
predictions

Orange.evaluation.CA(predictions)






## Majority learner sample
database = Orange.data.Table('/path/to/database') ## Importing database

database.domain ## Visualizing the values

majority = Orange.classification.MajorityLearner() ## Renaming function

predictions = Orange.evaluation.testing.TestOnTestData(database, database, [majority])

Orange.evaluation.CA(predictions) ## evaluation of the algorithm

for entry in database: ## Prints values collected from database, will print several th
print(entry.get_class())

from collections import Counter
Counter(str(entry.get_class()) for entry in database) ## Counts the values as an alter

"value 1" / "value 2" ## Evaluates the algorithm manually






## KNN sample

import pickle
with open('path/to/database.pk1', 'rb') as f:
x_database_train, y_database_train, x_database_test, y_database_teste = pickle.load(

x_database_train.shape, y_database_train.shape ## Visualizing

x_database_test.shape, y_database_test.shape

knn_function = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
knn_function.fit(x_database_train.shape, y_database_train.shape)

from sklearn.metrics import accuracy_score, classification_report ## Evaluating the al
accuracy_score(y_database_test, predictions)

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix())
cm.fit(x_database_train, y_database_train)
cm.score(x_database_test, y_database_test)







## Logistic Regression
from sklearn.linear_model import LogisticRegression

import pickle
with open('path/to/database.pk1', 'rb') as f:
x_database_train, y_database_train, x_database_test, y_database_teste = pickle.load(




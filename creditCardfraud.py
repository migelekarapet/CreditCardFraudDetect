# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

#uncomment the line below and provide the actual path to source file
#df = pd.read_csv(‘.../creditcard.csv’)

#let's see how the df looks like
df.head()

print(df.shape)
print(df.describe())

# let's have a look at how the abnormality distribution looks like
features = df.iloc[:,0:28].columns
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, c in enumerate(df[features]):
 ax = plt.subplot(gs[i])
 sns.distplot(df[c][df.Class == 1], bins=50)
 sns.distplot(df[c][df.Class == 0], bins=50)
 ax.set_xlabel(‘’)
 ax.set_title(‘hist of parameters: ‘ + str(c))
plt.show()


#let's see the number of fradulant cases in our dataframe

Fraud = df[df[‘Class’] == 1]
Valid = df[df[‘Class’] == 0]
outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)
print(‘Fraud Cases: {}’.format(len(df[df[‘Class’] == 1])))
print(‘Valid Transactions: {}’.format(len(df[df[‘Class’] == 0])))

print(“Amount details of malicious transactions”)
Fraud.Amount.describe()

print(“details of valid transactions”)
Valid.Amount.describe()

# Graph the correl. matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


#let's divide the X and the Y from the dataset
X=data.drop([‘Class’], axis=1)
Y=data[“Class”]
print(X.shape)
print(Y.shape)

X_data=X.values
Y_data=Y.values

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 42)

#Building another model/classifier ISOLATION FOREST
from sklearn.ensemble import IsolationForest
ifc=IsolationForest(max_samples=len(X_train),
 contamination=outlier_fraction,random_state=1)
ifc.fit(X_train)
scores_pred = ifc.decision_function(X_train)
y_pred = ifc.predict(X_test)

# Reshape the prediction values to 0 for valid, 1 for fraud. 
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1
n_errors = (y_pred != Y_test).sum()

#printing the confusion matrix
LABELS = [‘Normal’, ‘Fraud’]
conf_matrix = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS,
 yticklabels=LABELS, annot=True, fmt=”d”);
plt.title(“Confusion matrix”)
plt.ylabel(‘True class’)
plt.xlabel(‘Predicted class’)
plt.show()


#evaluation of the model
#printing every score of the classifier
#scoring in any thing
from sklearn.metrics import confusion_matrix
n_outliers = len(Fraud)
print(“the Model used is {}”.format(“Isolation Forest”))
acc= accuracy_score(Y_test,y_pred)
print(“The accuracy is {}”.format(acc))
prec= precision_score(Y_test,y_pred)
print(“The precision is {}”.format(prec))
rec= recall_score(Y_test,y_pred)
print(“The recall is {}”.format(rec))
f1= f1_score(Y_test,y_pred)
print(“The F1-Score is {}”.format(f1))
MCC=matthews_corrcoef(Y_test,y_pred)
print(“The Matthews correlation coefficient is{}”.format(MCC))

# Building the Random Forest Classifier (RANDOM FOREST)
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)
# predictions
y_pred = rfc.predict(X_test)

#Evaluating the classifier
#printing every score of the classifier
#scoring in any thing
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix
n_outliers = len(Fraud)
n_errors = (y_pred != Y_test).sum()
print(“The model used is Random Forest classifier”)
acc= accuracy_score(Y_test,y_pred)
print(“The accuracy is {}”.format(acc))
prec= precision_score(Y_test,y_pred)
print(“The precision is {}”.format(prec))
rec= recall_score(Y_test,y_pred)
print(“The recall is {}”.format(rec))
f1= f1_score(Y_test,y_pred)
print(“The F1-Score is {}”.format(f1))
MCC=matthews_corrcoef(Y_test,y_pred)
print(“The Matthews correlation coefficient is{}”.format(MCC))

#printing the confusion matrix
LABELS = [‘Normal’, ‘Fraud’]
conf_matrix = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt=”d”);
plt.title(“Confusion matrix”)
plt.ylabel(‘True class’)
plt.xlabel(‘Predicted class’)
plt.show()

#visualizing the random tree 
feature_list = list(X.columns)
# Import tools needed for visualization
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydot
#pulling out one tree from the forest
tree = rfc.estimators_[5]
export_graphviz(tree, out_file = ‘tree.dot’, feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file(‘tree.dot’)
# Write graph to a png file
display(Image(graph.create_png()))


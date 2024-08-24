'''
The following python script has been developed as part of a training course in Machine Learning with Python.
The intent of this script is to build various machine learning classification algorithms and analyze their performance on a specific dataset.

The dataset contains weather data of Sydney, Australia, and has extra columns like 'RainToday' and our target is 'RainTomorrow'. Based on historical data we aim a predicting
whether or not the next day is going to rain or not.

Copyright: @gabrielerutigliano

email: rutiglianogabriele@icloud.com
github: github.com/rutiglianogabriele
linkedin: linkedin.com/in/gabrielerutigliano
'''

# Importing the required libraries

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
import itertools

# Defining a function to generate a confusion matrix with only passying the predictions and the actual data
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Importing the dataset
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv"
df = pd.read_csv(filepath)

'''
Data pre-processing
''' 

# Lets have a look at the dataset
df.head()

# The dataset has some categorical variables which can be hardly intepreted, thus we convert them to binary
df_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])

# Next, we replace the values of the 'RainTomorrow' column changing them from a categorical column to a binary column. 
# We do not use the `get_dummies` method because we would end up with two columns for 'RainTomorrow' and we do not want, since 'RainTomorrow' is our target.
df_processed.replace(['No', 'Yes'], [0,1], inplace=True)

# Drop the useless date column
df_processed.drop('Date',axis=1,inplace=True)

### Lets split the dataset into features (X) and target variable (y) and into training and test set

# Convert all variables to float to be easily handled by sklearn
df_processed = df_processed.astype(float)

# Drop the target and assign it to a different variable
X = df_processed.drop(columns='RainTomorrow', axis=1)
Y = df_processed['RainTomorrow']

# Split into training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)


'''
KNN
''' 
# Generate the KNN model
KNN = KNeighborsClassifier(n_neighbors = 3)

# Train the model using the training data
KNN.fit(x_train, y_train)

# Lets make predictions
y_hat = KNN.predict(x_test)

# Lets have a look at some metrics
KNN_Accuracy_Score = accuracy_score(y_test, y_hat)
KNN_JaccardIndex = jaccard_score(y_test, y_hat)
KNN_F1_Score = f1_score(y_test, y_hat)

# Create a DataFrame to display the metrics
report = pd.DataFrame({
    'Metric': ['Accuracy Score', 'Jaccard Index', 'F1 Score'],
    'Value': [KNN_Accuracy_Score, KNN_JaccardIndex, KNN_F1_Score]
})

print(report)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_hat)
np.set_printoptions(precision=2)

print (classification_report(y_test, y_hat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['No','Yes'],normalize= False,  title='Confusion matrix')

'''
Decision Tree
''' 
# Create the Decision Tree model
Tree = DecisionTreeClassifier(random_state=10)  # You can set a random_state for reproducibility

# Train the model using the training data
Tree.fit(x_train, y_train)

# Lets make predictions
y_hat = Tree.predict(x_test)

# Lets have a look at some metrics
Tree_Accuracy_Score = accuracy_score(y_test, y_hat)
Tree_JaccardIndex = jaccard_score(y_test, y_hat)
Tree_F1_Score = f1_score(y_test, y_hat)

# Create a DataFrame to display the metrics
report = pd.DataFrame({
    'Metric': ['Accuracy Score', 'Jaccard Index', 'F1 Score'],
    'Value': [Tree_Accuracy_Score, Tree_JaccardIndex, Tree_F1_Score]
})

print(report)

# Create a DataFrame to display the metrics
report = pd.DataFrame({
    'Metric': ['Accuracy Score', 'Jaccard Index', 'F1 Score'],
    'Value': [Tree_Accuracy_Score, Tree_JaccardIndex, Tree_F1_Score]
})

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_hat)
np.set_printoptions(precision=2)

print (classification_report(y_test, y_hat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['No','Yes'],normalize= False,  title='Confusion matrix')


'''
Logistic regression
''' 
LR = LogisticRegression(solver='liblinear', random_state=1)

# Train the model using the training data
LR.fit(x_train, y_train)


y_hat = LR.predict(x_test)

predict_proba = LR.predict_proba(x_test)

LR_Accuracy_Score = accuracy_score(y_test, y_hat)
LR_JaccardIndex = jaccard_score(y_test, y_hat)
LR_F1_Score = f1_score(y_test, y_hat)
LR_Log_Loss = log_loss(y_test, predict_proba)

# Create a DataFrame to display the metrics
report = pd.DataFrame({
    'Metric': ['Accuracy Score', 'Jaccard Index', 'F1 Score', 'Log Loss'],
    'Value': [LR_Accuracy_Score, LR_JaccardIndex, LR_Log_Loss, LR_Log_Loss]
})

print(report)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_hat)
np.set_printoptions(precision=2)

print (classification_report(y_test, y_hat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['No','Yes'],normalize= False,  title='Confusion matrix')


'''
SVM
''' 
SVM = svm.SVC(kernel='rbf')

# Train the model using the training data
SVM.fit(x_train, y_train)

y_hat = SVM.predict(x_test)

SVM_Accuracy_Score = accuracy_score(y_test, y_hat)
SVM_JaccardIndex = jaccard_score(y_test, y_hat)
SVM_F1_Score = f1_score(y_test, y_hat)

# Create a DataFrame to display the metrics
report = pd.DataFrame({
    'Metric': ['Accuracy Score', 'Jaccard Index', 'F1 Score'],
    'Value': [SVM_Accuracy_Score, SVM_JaccardIndex, SVM_F1_Score]
})

print(report)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_hat)
np.set_printoptions(precision=2)

print (classification_report(y_test, y_hat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['No','Yes'],normalize= False,  title='Confusion matferix')





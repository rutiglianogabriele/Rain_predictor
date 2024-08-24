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
KNN = KNeighborsClassifier(n_neighbors=4)

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

# Lets print the confusion matrix
generate_confusion_matrix(y_hat, y_test) 

'''
Decision Tree
''' 
# Example data: true labels and predicted labels
y_true = [0, 1, 0, 1, 0, 1, 1, 0, 2, 2]
y_pred = [0, 0, 0, 1, 0, 1, 1, 1, 2, 1]

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=[f'Predicted {i}' for i in range(cm.shape[1])], 
            yticklabels=[f'Actual {i}' for i in range(cm.shape[0])])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


'''
Logistic regression
''' 



'''
SVM
''' 
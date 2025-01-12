import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve

data_file = os.path.abspath(r'C:\Users\Nitheesh\Downloads\data.csv')
data = pd.read_csv(data_file)

print('============================Data====================================')
print('\n')
print(data)

data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
print('==============================Data Info=============================')
print('\n')
print(data.info())
print('==============================Data describe==========================')
print('\n')
print(data.describe().T)
print('\n')

data['diagnosis'].replace({"M": "1", "B": "0"}, inplace=True)
data['diagnosis'] = data['diagnosis'].astype(int)  # Convert diagnosis to integer

# Define a color palette (you can customize this list of colors)
colors = ["#1f77b4", "#ff7f0e"]

# Create the plot with different colors for each bar
ax = sns.countplot(x='diagnosis', data=data, palette=colors)
# Add count on each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height()}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

# Show the plot
plt.show()

plt.figure(figsize=(20, 9))
sns.heatmap(data.corr(), annot=True, cmap="mako")

#==============Test Train=================
X = data.drop(['diagnosis'], axis=1)
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.fit_transform(X_test)
# Function to plot AUC-ROC curve
def plot_roc_curve(y_test, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')


# Function to plot confusion matrix heatmap
def plot_confusion_matrix_heatmap(cm, model_name):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Predicted B', 'Predicted M'], yticklabels=['Actual B', 'Actual M'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()


#==============Logistic Regression=================
model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print('===========Logistic Regression==============')

# Display confusion_matrix
cm = confusion_matrix(y_test, pred)
print("\nConfusion_matrix:")
print(cm)
plot_confusion_matrix_heatmap(cm, 'Logistic Regression')

# Evaluate the model
accuracy = accuracy_score(y_test, pred) * 100
print("\nAccuracy:", accuracy)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))
#==============KNN classfier model=================
knn = KNeighborsClassifier(n_neighbors=2)    # try k = 2 , 3 , 4

# Fitting the training data
knn.fit(X_train, y_train)

# Predicting on the test data
pred1 = knn.predict(X_test)
proba1 = knn.predict_proba(X_test)[:, 1]

print('===========KNN classfier==============')

# Display confusion_matrix
cm = confusion_matrix(y_test, pred1)
print("\nConfusion_matrix:")
print(cm)
plot_confusion_matrix_heatmap(cm, 'KNN Classifier')

# Evaluate the model
accuracy = accuracy_score(y_test, pred1) * 100
print("\nAccuracy:", accuracy)
# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, pred1))
#==============Random Forest Classifier=================
RF = RandomForestClassifier(n_estimators=100, random_state=0)

# Fitting the training data
RF.fit(X_train, y_train)

# Predicting on the test data
pred2 = RF.predict(X_test)
proba2 = RF.predict_proba(X_test)[:, 1]

print('===========Random Forest Classifier ==============')

# Display confusion_matrix
cm = confusion_matrix(y_test, pred2)
print("\nConfusion_matrix:")
print(cm)
plot_confusion_matrix_heatmap(cm, 'Random Forest')

# Evaluate the model
accuracy = accuracy_score(y_test, pred2) * 100
print("\nAccuracy:", accuracy)
# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, pred2))
#==============Decision Tree Classifier=================
DT = DecisionTreeClassifier(random_state=42)
DT.fit(X_train, y_train)
# Predicting on the test data
pred3 = DT.predict(X_test)
proba3 = DT.predict_proba(X_test)[:, 1]

print('===========Decision Tree Classifier==============')

# Display confusion_matrix
cm = confusion_matrix(y_test, pred3)
print("\nConfusion_matrix:")
print(cm)
plot_confusion_matrix_heatmap(cm, 'Decision Tree')

# Evaluate the model
accuracy = accuracy_score(y_test, pred3) * 100
print("\nAccuracy:", accuracy)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, pred3))

# Plot ROC curve
plot_roc_curve(y_test, proba, 'Logistic Regression')
plot_roc_curve(y_test, proba1, 'KNN Classifier')
plot_roc_curve(y_test, proba2, 'Random Forest')
plot_roc_curve(y_test, proba3, 'Decision Tree')

#==========================================================
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
#=============================================================

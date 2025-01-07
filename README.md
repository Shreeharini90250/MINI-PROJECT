# MINI-PROJECT
IMPLEMENTATION 
Initialization
!pip install seaborn –U 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns. version from google.colab import files uploaded = files.upload() 
data pd.read_csv('/content/Student_Mental_health.csv') df_clean = data.copy()
 df_clean.head(5)

Data Cleaning
# Check dtypes
 print('Col types:\n',df_clean.dtypes,'\n','='*25,sep='') 
# Check for NA values
 print('Number of NA per Col:') 
df_clean.isna().sum() 
# Since only age has NA we can replace it with the mean as an int df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].mean()).
astype('int64') df_clean.isna().sum() 
# Rename columns for clarity 
df_clean.rename(columns={ 
'Choose your gender':'Gender',
 'What is your course?':'Course',
 'Your current year of Study':'Year', 
 'What is your CGPA?':'GPA',
 'Maritalstatus':'Married', 
'Do you have Depression?':'Depression',
 'Do you have Anxiety?':'Anxiety', 
'Do you have Panic attack?':'Panic_Attacks',
 'Did you seek any specialist for a treatment?':'Treatment'}, inplace=True)

Gender to Responses & Conditions
 # Compare Gender to Mental Health conditions
sns.countplot(data=df_clean, hue='Anxiety', x='Gender', hue_order=['Yes','No']) plt.title('Anxiety by Gender') 
plt.xlabel('')
plt.show() 
sns.countplot(data=df_clean, hue='Depression', x='Gender', hue_order=['Yes','No'])
 plt.title('Depression by Gender')
 plt.xlabel('')
 plt.show()
 sns.countplot(data=df_clean, hue='Panic_Attacks', x='Gender', hue_order=['Yes','No'])
 plt.title('Panic Attacks by Gender') 
plt.xlabel('') 
plt.show()

Course to Conditions with Gender
# Compare courses to Anxiety and Gender
plt.figure(figsize=(10, 10)) 
sns.swarmplot(data=df_clean, x='Anxiety', y='Course', hue='Gender',order=['Yes','No'])
 plt.show()
 # Compare Courses to Depression and Gender
plt.figure(figsize=(10, 10)) 
sns.swarmplot(data=df_clean, x='Depression', y='Course', hue='Gender',order=['Yes','No'])
plt.show(
 # Compare Courses to Panic Attacks and Gender 
plt.figure(figsize=(10, 10))
sns.swarmplot(data=df_clean, x='Panic_Attacks', y='Course', hue='Gender',order=['Yes','No'])
 plt show()

Modelling
for col in df_clean.columns: print(df_clean[col].value_counts().sort_index(),'\n','='*50,sep='')

Data Pre-Processing
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble 
import RandomForestClassifier 
from sklearn.model_selection
import train_test_split, cross_val_score 
from sklearn.metrics 
import classification_report,accuracy_score 
from sklearn.preprocessing 
import StandardScaler 
df_model = df_clean.copy() df_model.drop(columns='Timestamp',inplace=True) 
df_model.dtypes 
#Convert Binary columns into numeric 
for col in bool_cols: 
df_model[col] = df_model[col].replace({'Yes':1,'No':0})

Train models 
# Split data 
X = df_model.drop(columns=['Depression']) 
y = df_model['Depression'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=89) 
# Decision Tree 
decision_tree = DecisionTreeClassifier(random_state=43) decision_tree.fit(X_train,y_train) 
print('Decision Tree:',cross_val_score(decision_tree, X_train, y_train, cv=8).mean()) 
# Random Forest 
random_forest = RandomForestClassifier(random_state=43) random_forest.fit(X_train,y_train) print('Random Forest:',cross_val_score(random_forest, X_train, y_train, cv=8).mean())

Decision tree and random forest accuracy 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics 
import accuracy_score, classification_report, confusion_matrix 
# Make predictions on the test set 
pred_dtree = decision_tree.predict(X_test) 
pred_rforest = random_forest.predict(X_test) 
# Calculate accuracy as percentage 
accuracy_dtree = accuracy_score(y_test, pred_dtree) * 100
accuracy_rforest = accuracy_score(y_test, pred_rforest) * 100 
# Print accuracies 
print(f'Decision Tree Accuracy: {accuracy_dtree:.2f}%') 
print(f'Random Forest Accuracy: {accuracy_rforest:.2f}%') 
# Print classification reports 
print('Decision Tree Classification Report:\n', classification_report(y_test, pred_dtree)) 
print('Random Forest Classification Report:\n', classification_report(y_test, pred_rforest)) 
# Compute confusion matrices 
cm_dtree = confusion_matrix(y_test, pred_dtree) 
cm_rforest = confusion_matrix(y_test, pred_rforest) 
# Plot confusion matrices 
fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
sns.heatmap(cm_dtree, annot=True, fmt='d', cmap='Blues', ax=axes[0]) axes[0].set_title('Decision Tree Confusion Matrix') 
axes[0].set_xlabel('Predicted Labels') 
axes[0].set_ylabel('True Labels') 
sns.heatmap(cm_rforest, annot=True, fmt='d', cmap='Greens', ax=axes[1]) axes[1].set_title('Random Forest Confusion Matrix') axes[1].set_xlabel('Predicted Labels') 
axes[1].set_ylabel('True Labels') 
plt.tight_layout() 
plt.show()

Comparison of algoritms 
import matplotlib.pyplot as plt 
# Define model names and accuracies 
models = ['Decision Tree', 'Random Forest'] 
accuracies = [accuracy_dtree, accuracy_rforest] 
# Plot the bar chart 
plt.figure(figsize=(5, 5)) 
bars = plt.bar(models, accuracies, color=['skyblue', 'orange']) 
# Annotate the accuracy percentages on top of each bar 
for bar, accuracy in zip(bars, accuracies): 
plt.text(bar.get_x() + bar.get_width() / 2, 
bar.get_height() - 5, f'{accuracy:.2f}%', ha='center', va='bottom', 
color='black', fontsize=12) 
# Add titles and labels 
plt.title('Comparison of Model Accuracies', fontsize=16) plt.ylabel
('Accuracy (%)', fontsize=12) plt.xlabel('Models', fontsize=12) 
plt.ylim(0, 100) 
# Set y-axis range to 0-100 for clarity 
# Highlight which model performs better 
better_model = models[accuracies.index(max(accuracies))] 
plt.text(0.5, 90, f'{better_model} has better accuracy!', ha='center', color='green', fontsize=14, bbox=dict(facecolor='white', edgecolor='green')) 
# Show the plot plt.tight_layout()

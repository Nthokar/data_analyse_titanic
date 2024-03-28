import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv')
df.info()
##stats
col = ['Sex', 'Fare', 'Age']

X = pd.DataFrame()
for i in col:
    if df[i].dtype.name != 'object':
        X[i] = df[i].copy()
        X.loc[X[i].isna(), i] = X[i].median()
    else:
        X[i] = pd.factorize(df[i])[0]

print(X)

Y = df['Survived'].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
model = RandomForestClassifier()
model.fit(x_train, y_train)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

ar_f=[]
for f, idx in enumerate(indices):
    ar_f.append([round(importances[idx],4), col[idx]])
print("Значимость признака:")
ar_f.sort(reverse=True)
print(ar_f)


from sklearn import metrics
# метрика, насколько точно мы предсказываем правильные значения как для 0, так и 1
print("Accuracy:",metrics.accuracy_score(y_test, model.predict(x_test)))


from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion_matrix(y_test, model.predict(x_test))

# так же матрица в процентах и более изящном виде
matrix = confusion_matrix(y_test, model.predict(x_test))
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['death', 'survived']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Предсказанные классы')
plt.ylabel('Истинные классы')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()
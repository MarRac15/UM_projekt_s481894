import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score

data = pd.read_csv("preprocessed_data.csv")
scaler = StandardScaler()

x = data.drop(['EngagementLevel'], axis=1)
y = data['EngagementLevel']

#podzial zbiorow:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=64)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#Uczenie:
model = LogisticRegression(penalty='l2', C=0.5, multi_class='multinomial', solver='lbfgs')
model.fit(x_train, y_train)

y_predicted = model.predict(x_test)
# EWALUACJA MODELU:

print()
print('Evaluation of the model:')

report_test = classification_report(y_test, y_predicted)
print(f'Metrics: \n{report_test}')
print('-'*80)
print()

prec, recall, fscore, support = precision_recall_fscore_support(y_test, y_predicted,average='weighted' ,pos_label=1)
accuracy = accuracy_score(y_test, y_predicted)
print(f"Precision equals: {prec}")
print(f"Recall equals: {recall}")
print(f"F-score equals: {fscore}")
print(f"Accuracy equals: {accuracy}")
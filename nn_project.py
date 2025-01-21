import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tensorflow import keras

# from keras.models import Sequential
# from keras.layers import Dense


data = pd.read_csv("preprocessed_data.csv")
scaler = StandardScaler()

x = data.drop(['EngagementLevel'], axis=1)
y = data['EngagementLevel']
x_array = x.to_numpy()
y_array = y.to_numpy(dtype='int32')

x_scaled = scaler.fit_transform(x_array)

# Podział:
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_array, test_size=0.2, stratify=y_array, random_state=64)
print (f'Shape of Train Data : {x_train.shape}')
print (f'Shape of Test Data : {x_test.shape}')

y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_test = keras.utils.to_categorical(y_test, num_classes=3)

# MODEL SIECI NEURONOWEJ:

num_classes = 3

model = keras.Sequential([
    keras.layers.Dense(64, activation = 'relu', input_dim = x_train.shape[1]),
    keras.layers.Dense(32, activation = 'relu'),
    keras.layers.Dense(num_classes, activation = 'softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

#TRENOWANIE:

model.fit(
    x_train,
    y_train,
    batch_size = 32,
    epochs = 100,
    verbose=1
)


# EWALUACJA MODELU:

print()
print('Evaluation of the model:')
y_predicted = model.predict(x_test)
y_predicted_classes = np.argmax(y_predicted, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

report_test = classification_report(y_test_classes, y_predicted_classes)
print(f'Metrics: \n{report_test}')
print('-'*80)


scores = model.evaluate(x_test, y_test, verbose=0)
print()
print("Test loss:", scores[0])
print()

prec, recall, fscore, support = precision_recall_fscore_support(y_test_classes, y_predicted_classes,average='weighted' ,pos_label=1)

print(f"Precision equals: {prec}")
print(f"Recall equals: {recall}")
print(f"F-score equals: {fscore}")
print(f'Accuracy: {round(scores[1]*100, 2)} %')
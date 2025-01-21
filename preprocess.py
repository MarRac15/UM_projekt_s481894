import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# W TYM PLIKU: ANALIZA DANYCH (EDA) I ZMIANY W ZBIORZE

data = pd.read_csv("online_gaming_behavior_dataset.csv")

print('---------------------------------------------')
print(f"Data shape: {data.shape}")
print()

#Czy są Nan:
print(f'Number of Nans in total: {data.isna().sum().sum()}') 
#0 Nans!

#Przegląd typów danych
print()
print('Data overview:\n ')
print(data.info())
#print(data.describe())

print()

#liczba wartości unikalnych / wszystkie wartosci w danej kolumnie:
print('The unique values vs all values in the column:\n ')
for col in data:
    unique_count = data[col].nunique()
    total_count = data[col].count()
    print(f"{col}: {unique_count}/{total_count}")



#Dane kategoryczne:
categorical = data.select_dtypes(include=['object']).columns
print()
print(f'Categorical columns: {categorical}')

#One hot encoding:
data = pd.get_dummies(data, columns=['Gender', 'Location', 'GameGenre', 'GameDifficulty'], dtype=int)


#Target column: Engagement level -> zmiana na numeryczne:
mapping = {'Low': 0, 'Medium': 1, 'High': 2}
data['EngagementLevel'] = data['EngagementLevel'].map(mapping)

#usuwam kolumne ID:

data = data.drop(['PlayerID'], axis=1)
print()
print(data)




#Sprawdzam rozklad klas:
print()
print("Target column distribution: ")
print(data['EngagementLevel'].value_counts())
data['EngagementLevel'].value_counts().plot(kind='bar', title='Class distribution')
plt.show()

#Klasa Medium ma wiecej przykladow, zbalansuje te klasy:
x = data.drop(['EngagementLevel'], axis=1)
y = data['EngagementLevel']

x_array = x.to_numpy()
y_array = y.to_numpy(dtype='int32')


# BALANSOWANIE KLAS:

low_indices = np.where(y_array == 0)[0]
medium_indices = np.where(y_array == 1)[0]
high_indices = np.where(y_array == 2)[0]
print(low_indices)

# kazda klasa bedzie miec po 10 000 probek
new_low_indices = low_indices[:10000]
new_medium_indices = medium_indices[:10000]
new_high_indices = high_indices[:10000]

new_indices = np.concatenate((new_low_indices, new_medium_indices, new_high_indices))
np.random.shuffle(new_indices)

# wybór wierszy po edycji:
data = data.iloc[new_indices]
print(data)
print(data['EngagementLevel'].value_counts())



#ZAPISUJE DANE DO OSOBNEGO PLIKU:
data.to_csv('preprocessed_data.csv', index=False)
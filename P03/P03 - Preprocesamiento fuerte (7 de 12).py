# Javier Rodríguez Rodríguez 78306251Z
# Inteligencia de Negocio - IN
# GII - UGR - 2021

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RS = 1410950

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
df = pd.concat((train, test), axis=0, ignore_index=True)
df = df.drop(['id', 'Nombre'], axis=1)


keys = df.keys().drop('Precio_cat')

# Exploración de datos
labels = df.iloc[:,-1].unique()
print('there are %d labels: \n'%len(labels))
labels_count = df.iloc[:,-1].value_counts()
print(labels_count)
labels_percentage = labels_count/np.sum(labels_count)*100
print(labels_percentage)

df['Descuento'] = df['Descuento'].fillna(0)

categoricas = ['Ciudad', 'Año', 'Combustible', 'Tipo_marchas', 'Mano', 'Asientos']
num_texto = ['Motor_CC', 'Consumo', 'Potencia']

# Preparación de datos de entrada
for i in num_texto:
    df[f"{i}"] = df[f"{i}"].str.extract(r'(\d+(.\d+)?)').astype(float)

# Eliminar columnas categóricas con clases muy poco representadas (<10)
for k in categoricas:
    if (df[k].value_counts().min() < 10):
        df =  df.drop(columns=k)
        keys = keys.drop(k) 


# Eliminar filas con NaN, o con 2 o más NaN
nulls = df[keys].isnull().sum()
overnan = df[keys].isnull().sum(axis=1)
# >1 si tolerancia a un NaN. >0 si no
print(overnan[overnan>0])
overnan = overnan[overnan>0].index
print(nulls[nulls!=0])

nulls_percentage = nulls[nulls!=0]/df.shape[0]*100
print('the percentages of null values per feature:\n')
print(round(nulls_percentage,2))

df = df.drop(overnan)
print(df)
print(f'{df.shape[0]}')

# Código para acceder a test
#print(df[df.isnull().sum(axis=1)>0])

# Análisis de correlación
sns.heatmap(df[keys].corr(), cmap='RdBu', center=0)
sns.clustermap(df[keys].corr(), cmap='RdBu', center=0)
plt.show()

# Kilómetros tiene casi nula correlación
df = df.drop(columns=['Kilometros']) # Descuentos también?
keys = keys.drop('Kilometros')

categoricas = ['Ciudad', 'Tipo_marchas']
print(df)

df = pd.get_dummies(df, columns=categoricas)
df.to_csv('preprocessed_dataset.csv')


########################### PARTE 2

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import SMOTE, RandomOverSampler


pd.set_option('display.max_columns', 5)
print(df)
train = df[df.isnull().sum(axis=1)==0]
test = df[df.isnull().sum(axis=1)==1].dropna(axis=1)
X = train.drop(columns=['Precio_cat'])
y = train.Precio_cat

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RS)

# Innecesario?
dfs = (X_train, X_test, y_train, y_test)
df_names = ('X_train', 'X_test', 'y_train', 'y_test')

# Probando que el split es bueno
def df_imbalance(df):
    labels = df.unique()
    print('there are %d labels: \n'%len(labels))
    labels_count = df.value_counts()
    print(labels_count)
    labels_percentage = labels_count/np.sum(labels_count)*100
    print(labels_percentage)

print('outputs of y_train:')
df_imbalance(y_train)
print('\noutputs of y_test:')
df_imbalance(y_test)

pipe = Pipeline([
    ('undersampler', 'passthrough'),
    ('oversampler', 'passthrough'),
    ('rf', RandomForestClassifier(random_state=RS, n_jobs=-1))
])

grid = {
    'undersampler': ['passthrough', RandomUnderSampler(random_state=RS), NearMiss(version=1)],
    'oversampler': ['passthrough', RandomOverSampler(random_state=RS), SMOTE(random_state=RS)],
    'rf__n_estimators': [10, 100, 500, 1000],
    'rf__criterion': ['gini', 'entropy'],
    'rf__max_depth': [100, 500, None],
    'rf__max_features': ['sqrt', 'log2', None],
    'rf__class_weight': ['balanced', 'balanced_subsample', None]
    #'n_estimators': [100],
    #'criterion': ['gini'],
    #'max_depth': [None],
    #'max_features': ['sqrt'],
    #'class_weight': ['balanced']

}

clf = GridSearchCV(pipe, param_grid=grid, n_jobs=-1, scoring='accuracy')
clf.fit(X, y)
df_table = pd.concat([pd.DataFrame(clf.cv_results_["params"]),pd.DataFrame(clf.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
df_table.to_csv("Tabla_rf.csv", index=False)

predict = clf.predict(test)
df_result = pd.DataFrame({'id': test.index, 'Precio_cat': predict})
df_result.to_csv("mis_resultados_rf.csv", index=False)


pipe = Pipeline([
    ('undersampler', 'passthrough'),
    ('oversampler', 'passthrough'),
    ('gb', GradientBoostingClassifier(random_state=RS))
])


grid = {
    'undersampler': ['passthrough', RandomUnderSampler(random_state=RS), NearMiss(version=1)],
    'oversampler': ['passthrough', RandomOverSampler(random_state=RS), SMOTE(random_state=RS)],
    'gb__loss': ['deviance', 'exponential'],
    'gb__learning_rate': [0.01, 0.1, 1, 10, 0.001],
    'gb__n_estimators': [50, 100, 500, 1000],
    'gb__criterion': ['friedman_mse', 'mse'],
    'gb__max_features': ['sqrt', 'log2', None],
    'gb__max_depth': [3, 5, 10, 20]
    #'n_estimators': [100],
    #'criterion': ['gini'],
    #'max_depth': [None],
    #'max_features': ['sqrt'],
    #'class_weight': ['balanced']

}

clf = GridSearchCV(pipe, param_grid=grid, n_jobs=-1, scoring='accuracy')
clf.fit(X, y)
df_table = pd.concat([pd.DataFrame(clf.cv_results_["params"]),pd.DataFrame(clf.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
df_table.to_csv("Tabla_gb.csv", index=False)

predict = clf.predict(test)
df_result = pd.DataFrame({'id': test.index, 'Precio_cat': predict})
df_result.to_csv("mis_resultados_gb.csv", index=False)


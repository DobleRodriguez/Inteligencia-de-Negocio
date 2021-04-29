# Javier Rodríguez Rodríguez 78306251Z
# Inteligencia de Negocio - IN
# GII - UGR - 2021

from os import pipe
from pathlib import Path
from platform import version
from typing import Counter
from urllib.request import UnknownHandler 
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from scipy.stats.stats import NormaltestResult
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedStratifiedKFold
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, Normalizer
from sklearn.naive_bayes import ComplementNB, GaussianNB, CategoricalNB
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, SMOTENC
from imblearn.under_sampling import ClusterCentroids, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier, BalancedBaggingClassifier, EasyEnsembleClassifier
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.metrics import geometric_mean_score

RS = 1410950

train = pd.read_csv(Path(__file__).parent / "train.csv")

# Empieza en 2 porque nos cargamos los nombres
train['Descuento'] = train['Descuento'].fillna(0)
train = train.dropna()
X = train.iloc[:,2:-1]
y = train.iloc[:,-1]

counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Clase=%s, Elementos=%s, Porcentaje=%.3f%%' % (k, v, per))
plt.bar(counter.keys(), counter.values())
plt.show()


"""

X[categoricas] = X[categoricas].fillna("missing")
X[categoricas] = OrdinalEncoder(categories=categorias, handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X[categoricas])

X[categoricas] = X[categoricas].replace(X[categoricas].max(), np.nan)

#print(X.loc[X["Motor_CC"] % 1 != 0 , enteras])
#print(X.loc[X["Ciudad"].isna(), categoricas])

#print(X[categoricas].iloc[30])
#print(X.loc[X["Ciudad"] == 'missing', categoricas])
X.dropna().hist(bins=20)
#plt.show()



"""

keys = X.keys()
categoricas = ['Ciudad', 'Combustible', 'Tipo_marchas', 'Mano']
enteras = ['Año', 'Kilometros', 'Motor_CC', 'Asientos']
reales = ['Consumo', 'Potencia', 'Descuento']
numericas = ['Año', 'Kilometros', 'Motor_CC', 'Asientos', 'Consumo', 'Potencia', 'Descuento']

num_texto = ['Motor_CC', 'Consumo', 'Potencia']
ord = ['Tipo_marchas', 'Mano']
oh = ['Ciudad', 'Combustible']

ordinales = []
for i in ord:
    ordinales.append(pd.read_csv(Path(__file__).parent / f"{i.lower()}.csv").to_numpy()[:,0])

onehot = []
for i in oh:
    onehot.append(pd.read_csv(Path(__file__).parent / f"{i.lower()}.csv").to_numpy()[:,0])


for i in num_texto:
    X[f"{i}"] = X[f"{i}"].str.extract('(\d+.?\d+)').astype(float)


iterative_disc_zero = ColumnTransformer(
    [
        ("catimputer", SimpleImputer(strategy='most_frequent'), [0, 3, 4, 5]),
        ("numimputer", IterativeImputer(initial_strategy='mean', random_state=RS), [1, 2, 6, 7, 8, 9]),
        ("discimputer", SimpleImputer(strategy='constant'), [10])       
    ]
)

encoder = ColumnTransformer(
    [
        ("ordinalencoder", OrdinalEncoder(categories=ordinales), [4, 5]),
        ("onehotencoder", OneHotEncoder(categories=onehot), [0, 3])
    ], remainder='passthrough'
)

#std_scaler = ColumnTransformer(
#    [
#        ("standardscaler", StandardScaler(), [7, 9, 11]),
#    ], remainder='passthrough'
#)

 
#print(DataFrame(X).iloc[30])
#X = iterative_disc_zero.fit_transform(X)
#print(DataFrame(X).iloc[30])
#X = encoder.fit_transform(X)
#print(DataFrame(X).iloc[30])
#exit()
#X = StandardScaler().fit_transform(X)
#print(DataFrame(X).iloc[30])
#exit()
#print(DataFrame(X))



pipeline = Pipeline([
    ('oversampling', 'passthrough'),
    ('encoder', 'passthrough'),
    ('undersampling', 'passthrough'),
    ('preprocessing1', 'passthrough'),
    #('preprocessing2', 'passthrough'),
    ('classifier', 'passthrough')
])

grid = [{
    'oversampling':
        [
            #RandomOverSampler(random_state=RS),
            #SMOTENC(categorical_features=[0, 1, 3, 4, 5]),
            'passthrough'
        ],
    'undersampling':
        [
            'passthrough',
            #RandomUnderSampler(random_state=RS)
        ],
    'encoder':
        [
            encoder
        ],
    'preprocessing1': 
        [
            #StandardScaler(),
            'passthrough'
        ],   
    'classifier': 
        [
            #RandomForestClassifier(random_state=RS),
            #KNeighborsClassifier(),
            #BalancedRandomForestClassifier(random_state=RS),
            #EasyEnsembleClassifier(random_state=RS),
            #RUSBoostClassifier(random_state=RS),
            #SVC(random_state=RS),
            #LogisticRegression(random_state=RS, n_jobs=-1),
            #RandomForestClassifier(random_state=RS),
            #GradientBoostingClassifier(random_state=RS),
            #BalancedRandomForestClassifier(random_state=RS),
            #StackingClassifier([('brf', BalancedRandomForestClassifier(random_state=RS, n_estimators=1000))]),
            #StackingClassifier([('brf', BalancedRandomForestClassifier(random_state=RS, n_estimators=1000))], GradientBoostingClassifier(random_state=RS)),
            #StackingClassifier([('brf', BalancedRandomForestClassifier(random_state=RS, n_estimators=1000))], MLPClassifier(random_state=RS, hidden_layer_sizes=[100]*5)),
            #StackingClassifier([('brf', BalancedRandomForestClassifier(random_state=RS, n_estimators=1000))], KNeighborsClassifier()),
            #RandomForestClassifier(random_state=RS, n_estimators=1000),
            #BalancedRandomForestClassifier(random_state=RS, n_estimators=1000),
            #OneVsRestClassifier(GradientBoostingClassifier(random_state=RS)),
            #OneVsOneClassifier(GradientBoostingClassifier(random_state=RS, n_estimators=1000)),
            #OneVsOneClassifier(RandomForestClassifier(random_state=RS, n_estimators=1000)),
            OneVsOneClassifier(BalancedRandomForestClassifier(random_state=RS, n_estimators=1000)),           
            StackingClassifier([('rs', OneVsOneClassifier(GradientBoostingClassifier(random_state=RS, n_estimators=1000)))], OneVsOneClassifier(BalancedRandomForestClassifier(random_state=RS, n_estimators=1000))),
            #StackingClassifier([('rs', OneVsOneClassifier(RandomForestClassifier(random_state=RS, n_estimators=1000)))], OneVsOneClassifier(GradientBoostingClassifier(random_state=RS, n_estimators=1000)))

            #OneVsRestClassifier(MLPClassifier(hidden_layer_sizes= [100]*5, random_state=RS)),
            #OneVsOneClassifier(MLPClassifier(hidden_layer_sizes= [100]*5, random_state=RS)),
            #OneVsRestClassifier(SVC(decision_function_shape='ovr', random_state=RS)),
            #OneVsOneClassifier(SVC(decision_function_shape='ovo', random_state=RS)),
            #RandomForestClassifier(random_state=RS, n_estimators=1000, min_samples_leaf=3),
            #RandomForestClassifier(random_state=RS, n_estimators=1000, criterion='entropy'),
            #LinearDiscriminantAnalysis(),
            ##QuadraticDiscriminantAnalysis(),
            #BalancedBaggingClassifier(random_state=RS),
        ]
},
#{
#    'imputer': [KNNImputer(), SimpleImputer(), IterativeImputer()],
#    'undersampling' : ['passthrough'],
#    'oversampling': [SMOTETomek(random_state=RS), SMOTEENN(random_state=RS)],
#    'classifier': 
#        [
#        DecisionTreeClassifier(random_state=RS)]
#}
]

cv = RepeatedStratifiedKFold(random_state=RS, n_repeats=1)

"""
################# DROPNA
train = pd.read_csv(Path(__file__).parent / "train.csv")
train["Descuento"] = train["Descuento"].fillna(0)
train = train.dropna()
X = train.iloc[:,1:-1]
y = train.iloc[:,-1]

# summarize the class distribution
categoricas = ['Nombre', 'Ciudad', 'Combustible', 'Tipo_marchas', 'Mano']
num_texto = ['Motor_CC', 'Consumo', 'Potencia']

categorias = []

for i in categoricas:
    categorias.append(np.concatenate(((pd.read_csv(Path(__file__).parent / f"{i.lower()}.csv").to_numpy()[:,0]), ['missing'])))


#print(X[categoricas].iloc[30])
#print(X.loc[X["Ciudad"] == 'missing', categoricas])
X[categoricas] = OrdinalEncoder(categories=categorias, handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X[categoricas])


for i in num_texto:
    X[f"{i}"] = X[f"{i}"].str.extract('(\d+)').astype(float)

######################################
""" 

clf = GridSearchCV(pipeline, param_grid=grid, cv=cv, n_jobs=-1, verbose=3, scoring='balanced_accuracy')
clf.fit(X, y)
df_table = pd.concat([pd.DataFrame(clf.cv_results_["params"]),pd.DataFrame(clf.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
df_table.to_csv("Tabla_stackings.csv", index=False)
print(f"{clf.best_estimator_} ------------------- {clf.best_score_}")

data_test = pd.read_csv(Path(__file__).parent / "test.csv")
data_test['Descuento'] = data_test['Descuento'].fillna(0)
ids = data_test['id']   
del data_test['id']
del data_test['Nombre']

for i in num_texto:
    data_test[f"{i}"] = data_test[f"{i}"].str.extract('(\d+.?\d+)').astype(float)

predict = clf.predict(data_test)
df_result = pd.DataFrame({'id': ids, 'Precio_cat': predict})
df_result.to_csv("mis_resultados.csv", index=False)
exit()

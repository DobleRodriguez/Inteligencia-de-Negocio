# Javier Rodríguez @doblerodriguez
# IN - GII - UGR 2020/2021

from pathlib import Path

import numpy as np
from numpy.core.multiarray import result_type
from numpy.core.numeric import cross
import pandas as pd
import seaborn as sns
from numpy.random import default_rng, random
from seaborn.axisgrid import Grid
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (KBinsDiscretizer, Normalizer,
                                   OrdinalEncoder, StandardScaler, LabelEncoder)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, make_scorer, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier

filename = "mamografias.csv"

def model_comparison(X, y):
    rng = 0
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1_score': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score)
    }

    knn = GridSearchCV(KNeighborsClassifier(), param_grid={'n_neighbors': [1, 2, 3, 5, 10], 'weights':['uniform', 'distance']}, n_jobs=-1).fit(X, y)
    log_regression = GridSearchCV(LogisticRegression(random_state=rng, max_iter=10000), param_grid={'C': [0.01, 0.1, 1, 10, 100]}, n_jobs=-1).fit(X, y)
    neural_net = GridSearchCV(MLPClassifier(random_state=rng, max_iter=10000), param_grid={'activation': ['logistic', 'relu']}, n_jobs=-1).fit(X, y)
    decision_tree = GridSearchCV(DecisionTreeClassifier(random_state=rng), param_grid={'criterion': ['gini', 'entropy'], 'max_features': ['auto', None]}, n_jobs=-1).fit(X, y)
    support_vector_machine = GridSearchCV(LinearSVC(random_state=rng, max_iter=10000), param_grid={'C': [0.01, 0.1, 1, 10, 100], 'loss': ['hinge', 'squared_hinge']}, n_jobs=-1).fit(X, y)

    print(knn.best_params_)
    print(log_regression.best_params_)
    print(neural_net.best_params_)
    print(decision_tree.best_params_)
    print(support_vector_machine.best_params_)

    dummy = cross_validate(DummyClassifier(random_state=rng), X, y, scoring=scoring)
    knn = cross_validate(knn.best_estimator_, X, y, scoring=scoring)
    log_regression = cross_validate(log_regression.best_estimator_, X, y, scoring=scoring)
    neural_net = cross_validate(neural_net.best_estimator_, X, y, scoring=scoring)
    decision_tree = cross_validate(decision_tree.best_estimator_, X, y, scoring=scoring)
    support_vector_machine = cross_validate(support_vector_machine.best_estimator_, X, y, scoring=scoring)
    naive_bayes = cross_validate(GaussianNB(), X, y, scoring=scoring)

    result_table = pd.DataFrame(
        {
            'dummy': [dummy['test_accuracy'].mean(),
                    dummy['test_precision'].mean(),
                    dummy['test_recall'].mean(),
                    dummy['test_f1_score'].mean(),
                    dummy['test_roc_auc'].mean()],
            'knn': [knn['test_accuracy'].mean(),
                    knn['test_precision'].mean(),
                    knn['test_recall'].mean(),
                    knn['test_f1_score'].mean(),
                    knn['test_roc_auc'].mean()],
            'log_regression': [log_regression['test_accuracy'].mean(),
                    log_regression['test_precision'].mean(),
                    log_regression['test_recall'].mean(),
                    log_regression['test_f1_score'].mean(),
                    log_regression['test_roc_auc'].mean()],
            'neural_net': [neural_net['test_accuracy'].mean(),
                    neural_net['test_precision'].mean(),
                    neural_net['test_recall'].mean(),
                    neural_net['test_f1_score'].mean(),
                    neural_net['test_roc_auc'].mean()],
            'decision_tree': [decision_tree['test_accuracy'].mean(),
                    decision_tree['test_precision'].mean(),
                    decision_tree['test_recall'].mean(),
                    decision_tree['test_f1_score'].mean(),
                    decision_tree['test_roc_auc'].mean()],
            'support_vector_machine': [support_vector_machine['test_accuracy'].mean(),
                    support_vector_machine['test_precision'].mean(),
                    support_vector_machine['test_recall'].mean(),
                    support_vector_machine['test_f1_score'].mean(),
                    support_vector_machine['test_roc_auc'].mean()],
            'naive_bayes': [naive_bayes['test_accuracy'].mean(),
                    naive_bayes['test_precision'].mean(),
                    naive_bayes['test_recall'].mean(),
                    naive_bayes['test_f1_score'].mean(),
                    naive_bayes['test_roc_auc'].mean()],
        }, index=['Accuracy', 'Precision', 'Recall', 'F1 score', 'ROC AUC']
    )
    result_table['Mejor resultado'] = result_table.idxmax(axis=1)
    return result_table


df = pd.read_csv(Path(__file__).parent / f"{filename}", na_values=['?'])
print(f"Datos totales {df.shape[0]}")
df = df.dropna()
print(f"Datos completos {df.shape[0]}")


unique, count = np.unique(df['Severity'], return_counts=True)
print(count)

sns.countplot(x="Severity", data=df)
X, y = df.iloc[:, :-1], df.loc[:, 'Severity']

y = LabelEncoder().fit_transform(y)

X['Shape'] = LabelEncoder().fit_transform(X['Shape'])

result_table = model_comparison(X, y)
print(result_table)

X = StandardScaler().fit_transform(X)
X = Normalizer().fit_transform(X)
result_table = model_comparison(X, y)
print(result_table)


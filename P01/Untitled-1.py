
    neural_net = cross_validate(MLPClassifier(), X, y, scoring=scoring)
    decision_tree = cross_validate(DecisionTreeClassifier(), X, y, scoring=scoring)
    supp_vector_machine = cross_validate(LinearSVC(), X, y, scoring=scoring)
    naive_bayes = cross_validate(GaussianNB(), X, y, scoring=scoring)

    print(knn)
    print(neural_net)
    print(decision_tree)
    print(supp_vector_machine)
    print(naive_bayes)

    knn = GridSearchCV(KNeighborsClassifier(), param_grid={'n_neighbors': [1, 2, 3, 5, 10], 'weights':['uniform', 'distance']}, n_jobs=-1, scoring=scoring, refit='roc_auc').fit(X, y)
    log_regression = GridSearchCV(LogisticRegression(random_state=rng), param_grid={'C': [0.01, 0.1, 1, 10, 100]}, n_jobs=-1).fit(X, y)
    neural_net = GridSearchCV(MLPClassifier(random_state=rng), param_grid={'activation': ['logistic', 'relu']}, n_jobs=-1).fit(X, y)
    decision_tree = GridSearchCV(DecisionTreeClassifier(random_state=rng), param_grid={'criterion': ['gini', 'entropy'], 'max_features': ['auto', None]}, n_jobs=-1).fit(X, y)

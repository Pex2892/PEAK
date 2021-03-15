import os
import pandas as pd
import numpy as np
from itertools import combinations, chain, product
import multiprocessing as mlp
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, KFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import math
import ast
import warnings
warnings.filterwarnings('ignore')

class Resampling:

    def automate_lm(self, dataset, colsY):
        l_colsY = colsY.split(',')
        print(f'Columns in Y: {l_colsY}')

        l_colsX = dataset[dataset.columns.difference(l_colsY)].select_dtypes(include=['int64', 'float64']).columns.values
        print(f'Columns in X: {l_colsX}')

        results = []
        # step-1: create a cross-validation scheme
        for n_split in range(2, 6, 1):
            print(f'N splits: {n_split}')
            kf = RepeatedKFold(n_splits=n_split, n_repeats=10, random_state=None)

            for train_index, test_index in kf.split(dataset[l_colsX]):
                X_train = dataset.loc[train_index, l_colsX]
                y_train = dataset.loc[train_index, l_colsY]

                # step-2: specify range of hyperparameters to tune
                hyper_params = [{'n_features_to_select': list(range(1, len(l_colsX)+1))}]

                # step-3: perform grid search
                # 3.1 specify model
                lm = LinearRegression(fit_intercept=True, positive=True)
                lm.fit(X_train, y_train)
                rfe = RFE(lm)

                # 3.2 call GridSearchCV()
                model_cv = GridSearchCV(estimator=rfe, param_grid=hyper_params, scoring='r2', cv=kf,
                                        return_train_score=True, n_jobs=-1)

                # fit the model
                model_cv.fit(X_train, y_train)

                # cv results
                cv_res = model_cv.cv_results_

                # plotting cv results
                # plt.figure(figsize=(15, 5))
                #
                # plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
                # plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
                # plt.xlabel('number of features')
                # plt.ylabel('r-squared')
                # plt.title("Optimal Number of Features")
                # plt.legend(['train score', 'test score'], loc='upper left')
                # plt.show()

                for nf, mtrain, mtest in zip(cv_res["param_n_features_to_select"], cv_res["mean_train_score"], cv_res["mean_test_score"]):
                    results.append({
                        'method': 'RepeatedKFold',
                        'n_splits': n_split,
                        'n_repeats': kf.n_repeats,
                        'cols_Y': colsY,
                        'n_features_Y': len(l_colsY),
                        'cols_X': ','.join(l_colsX),
                        'n_features_X': nf,
                        'train_size': len(train_index),
                        'test_size': len(test_index),
                        'mean_train_score': mtrain,
                        'mean_test_score': mtest
                    })
        df_cv_results = pd.DataFrame(results)
        df_cv_results.to_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'CV_combinations_features_regression.csv'),
                             index=False, header=True, sep='\t', encoding='utf-8')
        print(f'The file "CV_combinations_features_regression.csv" has been saved')
        print(f'{"-" * 25}')

        df = pd.read_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'CV_combinations_features_regression.csv'), sep='\t')
        # print(df)

        df_group = df.groupby(by=['n_splits', 'n_features_X'])
        rows = []
        for name, group in df_group:
            idx = group['mean_test_score'].idxmax()
            row = df.iloc[idx, :]
            if row['mean_test_score'] > 0.0:
                rows.append(row.to_dict())

        df_best_CV = pd.DataFrame(rows, columns=df.columns)
        df_best_CV.to_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'CV_best_features_regression.csv'), index=False,
                          header=True, sep='\t', encoding='utf-8')
        print(f'The file "CV_best_features_regression.csv" has been saved')
        print(f'{"-" * 25}')


    def automate_classifiers(self, dataset, colsY):
        # example of grid searching key hyperparametres for logistic regression
        # define dataset

        Y = dataset[[f'{colsY}_fact']]
        print(f'Columns in Y: {Y.columns.values}')

        X = dataset[dataset.columns.difference([colsY, f'{colsY}_fact'])].select_dtypes(include=['int64', 'float64'])
        print(f'Columns in X: {X.columns.values}')

        '''models = [
            [LogisticRegression(), dict(solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], penalty=['l1', 'l2', 'elasticnet', 'none'], C=[100, 10, 1.0, 0.1, 0.01])],
            [KNeighborsClassifier(), dict(n_neighbors=range(1, 21, 2), weights=['uniform', 'distance'], algorithm=['ball_tree', 'kd_tree', 'brute'], metric=['euclidean', 'manhattan', 'minkowski'])],
            [SVC(), dict(kernel=['linear', 'poly', 'rbf', 'sigmoid'], C=[50, 10, 1.0, 0.1, 0.01], gamma=['scale'])],
            [RandomForestClassifier(), dict(n_estimators=[10, 100, 1000], criterion=['gini', 'entropy'], max_features=['sqrt', 'log2', 'none'], class_weight=['balanced', 'balanced_subsample', 'none'])],
            [GaussianNB(), dict(var_smoothing=np.logspace(0, -9, num=100))],
            [DecisionTreeClassifier(), dict(criterion=['gini', 'entropy'], splitter=['best', 'random'], max_features=['sqrt', 'log2', 'none'], class_weight=['balanced', 'none'])],
            [MLPClassifier(), dict(hidden_layer_sizes=[(50, 50, 50), (50, 100, 50), (100,)], activation=['identity', 'logistic', 'tanh', 'relu'], solver=['lbfgs', 'sgd', 'adam'], alpha=[0.0001, 0.05], learning_rate=['constant', 'invscaling', 'adaptive'], max_iter=[100, 200])]
        ]
        cv = [RepeatedStratifiedKFold(n_splits=i, n_repeats=2, random_state=2892) for i in range(2, 5, 1)]

        combs = list(product(*[models, cv]))
        for i in range(0, len(combs)):
            self._hyperparams_classifiers(Y, X, combs[i][0][0], combs[i][0][1], combs[i][1])'''


        # df_cv_results = pd.DataFrame(best_rows)
        # df_cv_results.to_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'CV_best_hyperparams_classification.csv'),
        #                     index=False, header=True, sep='\t', encoding='utf-8')

        # test come prendere i dati dal csv
        '''best_params = "{'C': 10, 'penalty': 'l2', 'solver': 'liblinear'}"
        best_params = ast.literal_eval(best_params)
        model = LogisticRegression()
        model.C = best_params['C']
        model.penalty = best_params['penalty']
        model.solver = best_params['solver']
        n_split = 3
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=math.ceil(X.shape[0]/n_split), random_state=2892)
        model = model.fit(X_train, Y_train)  # passiamo i set di addestramento
        probas_ = model.predict_proba(X_test)

        Y_pred = model.predict(X_test)  # eseguiamo la predizione sul test set

        tn, fp, fn, tp = sm.confusion_matrix(Y_test, Y_pred).ravel()

        # Posso sapere tutte le caratteristiche del classificatore
        # print(sm.classification_report(Y_test, Y_pred))

        fpr, tpr, thresholds = sm.roc_curve(Y_test, probas_[:, 1])

        scores = [
            sm.accuracy_score(Y_test, Y_pred),
            sm.precision_score(Y_test, Y_pred),
            sm.recall_score(Y_test, Y_pred),
            sm.f1_score(Y_test, Y_pred),
            sm.auc(fpr, tpr)
        ]
        print(scores)'''


    def _hyperparams_classifiers(self, Y, X, model, grid, cv, scoring: str = 'accuracy'):
        gs = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=scoring, error_score=0)
        grid_result = gs.fit(X, Y)

        rows = []
        for i in range(0, grid_result.cv_results_['mean_test_score'].shape[0], 1):
            rows.append({
                'model': gs.estimator,
                'X': ','.join(X.columns.values),
                'Y': ','.join(Y.columns.values),
                'cv': cv.__str__().split('(')[0],
                'n_splits': int(cv.get_n_splits(X) / cv.n_repeats),
                'n_repeats': cv.n_repeats,
                'random_state': cv.random_state,
                'scoring': scoring,
                'mean_test_score': grid_result.cv_results_['mean_test_score'][i],
                'std_test_score': grid_result.cv_results_['std_test_score'][i],
                'parameters': grid_result.cv_results_['params'][i]
            })

        # best rows
        # rows[grid_result.best_index_]

        df_cv_results = pd.DataFrame(rows)
        p = os.path.join(os.getcwd(), 'results', 'cross_validation', 'CV_hyperparams_classification.csv')
        df_cv_results.to_csv(p, mode='a', index=False, header=not os.path.exists(p), sep='\t', encoding='utf-8')


        # example of grid searching key hyperparametres for SVC
        # define dataset
        # define model and parameters


        # example of grid searching key hyperparameters for RandomForestClassifier
        '''from sklearn.ensemble import RandomForestClassifier
        # define dataset
        # define models and parameters
        model = RandomForestClassifier()
        n_estimators = [10, 100, 1000]
        criterion = ['gini', 'entropy']
        max_features = ['auto', 'sqrt', 'log2']
        class_weight = ['balanced', 'balanced_subsample', None]
        # define grid search
        grid = dict(n_estimators=n_estimators, criterion=criterion, max_features=max_features, class_weight=class_weight)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=2892)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
                                   error_score=0)
        grid_result = grid_search.fit(X, Y)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))'''






    '''def automate(self, dataset, colsY):
        l_colsY = colsY.split(',')
        print(f'Columns in Y: {l_colsY}')
        combs_colsY = [list(combinations(l_colsY, i)) for i in range(1, len(l_colsY)+1)]
        combs_colsY = list(chain(*combs_colsY))  # flatten list
        combs_colsY = list(map(list, combs_colsY))  # convert tuple to list
        print(f'Number of combinations of Y: {len(combs_colsY)}')
        print(f'{"-" * 25}')

        # dalle colonne del dataset escludo quelle passate come y_regr e seleziona solo quelle numeriche
        l_colsX = dataset[dataset.columns.difference(l_colsY)].select_dtypes(include=['int64', 'float64']).columns.values
        print(f'Columns in X: {l_colsX}')
        combs_colsX = [list(combinations(l_colsX, i)) for i in range(1, 3)] # qui bisogna mettere la len(colsX)+1
        combs_colsX = list(chain(*combs_colsX))  # flatten list
        combs_colsX = list(map(list, combs_colsX))  # convert tuple to list
        print(f'Number of combinations of X: {len(combs_colsX)}')
        print(f'{"-" * 25}')

        # definisce il max_k cosi da definire quanti split bisogna creare
        max_k = 5
        if dataset.shape[0] > 250:
            max_k = 3
        l_kfold = [RepeatedKFold(n_splits=k, n_repeats=10, random_state=None) for k in range(2, max_k+1, 1)]
        print(f'Number of combinations of Repeated K-fold: {len(l_kfold)}')
        print(f'{"-" * 25}')

        # creo la lista di tutte le possibili combinazioni delle 3 liste
        combs = list(product(*[combs_colsY, combs_colsX, l_kfold]))
        print(f'Number of total combinations: {len(combs)}')
        print(f'{"-" * 25}')

        model = LinearRegression()
        r = Parallel(n_jobs=mlp.cpu_count(), verbose=10)(delayed(self._evaluate_CV)(model, dataset[c[0]], dataset[c[1]], c[2]) for c in combs)
        r = list(chain(*r))  # flatten list

        df_comb = pd.DataFrame(r, columns=['X_cols', 'Y_cols', 'train_index', 'train_size', 'test_index',
                                           'test_size', 'cv_method', 'n_splits', 'n_repeats', 'R2_score', 'Adjusted_R2_score'])
        df_comb.to_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'combinations_CV.csv'),
                       index=False, header=True, sep='\t', encoding='utf-8')
        print(f'The file "combinations_CV.csv" has been saved')
        print(f'{"-" * 25}')

        self._best_CV()
        print(f'The file "best_CV.csv" has been saved')
        print(f'{"-" * 25}')

    def _evaluate_CV(self, model, Y, X, kf):
        rows = []
        for train_index, test_index in kf.split(X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
            y_train, y_test = Y.loc[train_index, :], Y.loc[test_index, :]

            model.fit(X_train, y_train)
            pred_values = model.predict(X_test)

            r2_score = sm.r2_score(y_test, pred_values)
            n_obs, n_regressors = X.shape
            adj_r2_score = 1 - (1 - r2_score) * (n_obs - 1) / (n_obs - n_regressors - 1)

            r = {
                'X_cols': ', '.join(X.columns.values),
                'Y_cols': ', '.join(Y.columns.values),
                'train_index': ",".join(map(str, train_index)),
                'train_size': len(train_index),
                'test_index': ",".join(map(str, test_index)),
                'test_size': len(test_index),
                'cv_method': 'RepeatedKFold',
                'n_splits': int(kf.get_n_splits(X) / kf.n_repeats),
                'n_repeats': kf.n_repeats,
                'R2_score': r2_score,
                'Adjusted_R2_score': adj_r2_score
            }
            rows.append(r)
        return rows

    def _best_CV(self):
        df = pd.read_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'combinations_CV.csv'), sep='\t')
        # print(df)

        df_group = df.groupby(by=['Y_cols', 'X_cols'])
        # print(df_group.groups)

        rows = []
        for name, group in df_group:
            # r2-negative
            # https://stackoverflow.com/questions/23036866/scikit-learn-is-returning-coefficient-of-determination-r2-values-less-than-1

            idx = group['Adjusted_R2_score'].idxmax()
            row = df.iloc[idx, :]
            rows.append(row.to_dict())
            # idx = group['Adjusted_R2_score'].idxmin()
            # row = df.iloc[idx, :]
            # rows.append(row.to_dict())

        df_best_CV = pd.DataFrame(rows, columns=df.columns)
        df_best_CV.to_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'best_CV.csv'), index=False,
                          header=True, sep='\t', encoding='utf-8')'''

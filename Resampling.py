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

        models = [
            [LogisticRegression(), dict(solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], penalty=['l1', 'l2', 'elasticnet', 'none'], C=[100, 10, 1.0, 0.1, 0.01])],
            [KNeighborsClassifier(), dict(n_neighbors=range(1, 21, 2), weights=['uniform', 'distance'], algorithm=['ball_tree', 'kd_tree', 'brute'], metric=['euclidean', 'manhattan', 'minkowski'])],
            [SVC(), dict(kernel=['linear', 'poly', 'rbf', 'sigmoid'], C=[50, 10, 1.0, 0.1, 0.01], gamma=['scale'])],
            [RandomForestClassifier(), dict(n_estimators=[10, 100, 1000], criterion=['gini', 'entropy'], max_features=['sqrt', 'log2', 'none'], class_weight=['balanced', 'balanced_subsample', 'none'])],
            [GaussianNB(), dict(var_smoothing=np.logspace(0, -9, num=100))],
            [DecisionTreeClassifier(), dict(criterion=['gini', 'entropy'], splitter=['best', 'random'], max_features=['sqrt', 'log2', 'none'], class_weight=['balanced', 'none'])],
            [MLPClassifier(), dict(hidden_layer_sizes=[(50, 50, 50), (50, 100, 50), (100,)], activation=['identity', 'logistic', 'tanh', 'relu'], solver=['lbfgs', 'sgd', 'adam'], alpha=[0.0001, 0.05], learning_rate=['constant', 'invscaling', 'adaptive'], max_iter=[100, 200])]
        ]
        cv = [RepeatedStratifiedKFold(n_splits=i, n_repeats=2, random_state=2892) for i in range(2, 3, 1)]

        combs = list(product(*[models, cv]))
        for i in range(0, len(combs)):
            self._hyperparams_classifiers(Y, X, combs[i][0][0], combs[i][0][1], combs[i][1])

        df_clf = pd.read_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'CV_hyperparams_classification.csv'), sep='\t')

        df_group = df_clf.groupby(by=['model'])
        idx = df_group['mean_test_score'].idxmax().to_list()
        df_clf.iloc[idx, :].to_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'CV_best_hyperparams_classification.csv'), index=False, header=True, sep='\t', encoding='utf-8')

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

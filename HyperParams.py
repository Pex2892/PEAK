import os
import math
import random
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import multiprocessing as mlp
from itertools import combinations, chain
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, LeaveOneOut, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
import sklearn.metrics as sm


class HyperParams:

    def best_split(self, df, colsX, colsY, max_splits=5, max_repeats=5, seed: int = 2021, filename: str = 'K-fold_TrainTestSplit.csv'):
        print(f'X columns: {colsX}')
        print(f'Y columns: {colsY}')
        print(f'{"-" * 25}')

        combs = [list(combinations(colsX, i)) for i in range(1, len(colsX)+1)]
        combs = list(chain(*combs))  # flatten list
        print(f'Combination of all columns: {combs}')
        print(f'{"-" * 25}')

        to_test = []
        for cols in combs:
            for k in range(2, max_splits):
                to_test.append([list(cols), RepeatedKFold(n_splits=k, n_repeats=max_repeats, random_state=None)])
        print(f'Tests to be processed: {len(to_test)}')
        print(f'{"-" * 25}')

        r = Parallel(n_jobs=mlp.cpu_count(), verbose=10)(delayed(self._evaluate_kfold)(df, t, colsY) for t in to_test)
        r = list(chain(*r))  # flatten list
        print(f'From {len(to_test)} tests, {len(r)} train and test split combinations were extracted for the dataset')
        print(f'{"-" * 25}')

        df_comb = pd.DataFrame(r, columns=['X_cols', 'Y_cols', 'train_index', 'test_index', 'k-fold',
                                           'k-fold_n_splits', 'k-fold_n_repeats', 'train_size', 'test_size',
                                           'R2_score'])
        df_comb.to_csv(os.path.join(os.getcwd(), 'results', filename), index=False,
                       header=True, sep='\t', encoding='utf-8')
        print(f'The file "{filename}" has been saved')
        print(f'{"-" * 25}')

    def _evaluate_kfold(self, df, t, colsY):
        model = LinearRegression()
        rows = []
        for train_index, test_index in t[1].split(df[t[0]]):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = df.loc[train_index, t[0]], df.loc[test_index, t[0]]
            y_train, y_test = df.loc[train_index, colsY], df.loc[test_index, colsY]

            model.fit(X_train, y_train)
            pred_values = model.predict(X_test)

            r = {
                'X_cols': ', '.join(t[0]),
                'Y_cols': ', '.join(colsY),
                'train_index': ",".join(map(str, train_index)),
                'test_index': ",".join(map(str, test_index)),
                'k-fold': 'RepeatedKFold',
                'k-fold_n_splits': int(t[1].get_n_splits(df[t[0]]) / t[1].n_repeats),
                'k-fold_n_repeats': t[1].n_repeats,
                'train_size': round(train_index.shape[0] / df.shape[0], 3),
                'test_size': round(test_index.shape[0] / df.shape[0], 3),
                'R2_score': sm.r2_score(y_test, pred_values)
            }
            rows.append(r)
        return rows

    def best_regression(self, df, colsX, colsY, is_comb: bool = False, max_random=5, filename: str = 'best_regression.csv'):
        print(f'X columns: {colsX}')
        print(f'Y columns: {colsY}')
        print(f'{"-" * 25}')

        range_testsize = np.arange(0.3, 0.51, 0.01)
        print(f'Range test size: {len(range_testsize)}')
        print(f'{"-" * 25}')

        to_test = []
        if is_comb:
            combs = [list(combinations(colsX, i)) for i in range(1, len(colsX)+1)]
            combs = list(chain(*combs))  # flatten list
            print(f'Combination of all columns: {combs}')
            print(f'{"-" * 25}')

            to_test = []
            for cols in combs:
                for size in range_testsize:
                    randomlist = random.sample(range(0, 100000), max_random)
                    for seed in randomlist:
                        to_test.append([list(cols), round(size, 2), seed])
        else:
            for size in range_testsize:
                randomlist = random.sample(range(0, 100000), max_random)
                for seed in randomlist:
                    to_test.append([colsX, round(size, 2), seed])

        print(f'Tests to be processed: {len(to_test)}')
        print(f'{"-" * 25}')

        r = Parallel(n_jobs=mlp.cpu_count(), verbose=100)(delayed(self._evaluate_regression)
                                                         (df, t, colsY) for t in to_test)
        df_comb = pd.DataFrame(r, columns=['X', 'Y', 'test_size', 'random_state', 'r2_score'])

        df_group = df_comb.groupby(['X'])

        r = Parallel(n_jobs=mlp.cpu_count(), verbose=10)(
            delayed(self._max_r2score_group)(df_group, group) for group in df_group.groups.keys())

        df_result = df_comb.iloc[r, :].sort_values(by=['r2_score'], ascending=False)

        df_result.to_csv(os.path.join(os.getcwd(), 'results', filename),
                         index=False, header=True, sep='\t', encoding='utf-8')
        print(f'The file "{filename}" has been saved')
        print(f'{"-" * 25}')

    def _evaluate_regression(self, df, t, colsY):
        X_train, X_test, Y_train, Y_test = train_test_split(df[t[0]], df[colsY], test_size=t[1], random_state=t[2], shuffle=True)
        model = LinearRegression(fit_intercept=True, normalize=True, positive=True)
        model = model.fit(X_train, Y_train)  # passiamo i set di addestramento

        Y_pred = model.predict(X_test)  # eseguiamo la predizione sul test set

        row = {
            'X': ','.join(t[0]),
            'Y': ','.join(colsY),
            'test_size': t[1],
            'random_state': t[2],
            'r2_score': sm.r2_score(Y_test, Y_pred)
        }
        return row

    def _max_r2score_group(self, df, group):
        return df.get_group(group)['r2_score'].idxmax()

    '''
    def for_classifiers(self, dataset, colsY_numeric):
        df = pd.read_csv(os.path.join(os.getcwd(), 'results', f'HyperParams_TrainTestSplit.csv'), sep='\t')
        df = df[df['r2_score'] > 0.20]
        # print(df)

        if df.shape[0] > 0:

            comb = []

            for i, r in df.iterrows():
                settings = [{
                    'colsX': r['X'].split(', '),
                    'model_name': 'Logistic Regression',
                    'model': LogisticRegression(),
                    'tuned_params': [{
                        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                        'C': [100, 10, 1.0, 0.1, 0.01]
                    }]
                }]
                comb.append(settings)

            print(comb)





            # for i, r in df.iterrows():
            #     self.best_logistic_regression(r, dataset, colsY_numeric)

        else:
            print('Non sono state trovati split buoni da poter eseguire l\'hyperparams per i classificatori')
            exit()

    def best_logistic_regression(self, r, dataset, colsY_numeric):
        colsX = r['X'].split(', ')
        print(colsX)
        print(dataset[colsX])
        X_train, X_test, Y_train, Y_test = train_test_split(dataset[colsX], dataset[colsY_numeric],
                                                            test_size=r['test_size'], random_state=r['random_state'],
                                                            shuffle=True)

        model = LogisticRegression()

        tuned_parameters = [{
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [100, 10, 1.0, 0.1, 0.01]
        }]

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        model_grid = GridSearchCV(estimator=model, param_grid=tuned_parameters, n_jobs=-1,
                                  cv=cv, scoring='accuracy', error_score=0)
        model_grid_res = model_grid.fit(X_train, Y_train)

        print("Best: %f using %s" % (model_grid_res.best_score_, model_grid_res.best_params_))
        means = model_grid_res.cv_results_['mean_test_score']
        stds = model_grid_res.cv_results_['std_test_score']
        params = model_grid_res.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))'''

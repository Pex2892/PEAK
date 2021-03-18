import os
import pandas as pd
import math
from itertools import combinations, chain
import multiprocessing as mlp
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
import matplotlib.pyplot as plt


class Regression:

    def __init__(self, ds, settings, init_settings):
        self.objDS = ds
        self.settings = settings
        self.init_settings = init_settings

    def automate(self):
        if self.settings['enable'] == 1:
            self._resampling()
            print('>>> Resamplig using linear regression has been completed')

            self.calculate()

    def _resampling(self):
        Y_cols = self.settings['y'].split(',')
        print(f'Columns in Y: {Y_cols}')

        X = self.objDS.dataset[self.objDS.dataset.columns.difference(Y_cols)].select_dtypes(include=['int64', 'float64'])
        print(f'Columns in X: {X.columns.values}')

        # step-1: create a cross-validation scheme
        cv = [RepeatedKFold(n_splits=i, n_repeats=self.settings['resampling']['n_repeats'],
                            random_state=self.init_settings['seed'])
              for i in range(self.settings['resampling']['min_split'], self.settings['resampling']['max_split'] + 1, 1)]
        print(f'Cross-Validation: {cv}')

        # step-2: specify range of hyperparameters to tune
        grid = dict(n_features_to_select=list(range(1, X.shape[1] + 1)))

        # step-3: perform grid search
        # 3.1 specify model
        lm = LinearRegression(fit_intercept=True, positive=True)
        rfe = RFE(lm)

        rows = []
        best_rows = []
        for i in range(0, len(cv)):
            # 3.2 call GridSearchCV()
            gs = GridSearchCV(estimator=rfe, param_grid=grid, scoring=self.settings['resampling']['scoring'], cv=cv[i],
                              return_train_score=True, n_jobs=self.init_settings['cpu'])

            # fit the model
            grid_result = gs.fit(X, self.objDS.dataset.loc[:, Y_cols])

            for j in range(0, grid_result.cv_results_['mean_test_score'].shape[0], 1):
                rows.append({
                    'model': gs.estimator,
                    'cv': cv[i].__str__().split('(')[0],
                    'n_splits': int(cv[i].get_n_splits(X) / cv[i].n_repeats),
                    'n_repeats': cv[i].n_repeats,
                    'random_state': cv[i].random_state,
                    'Y': ','.join(Y_cols),
                    'n_features_Y': len(Y_cols),
                    'n_features_X': grid_result.cv_results_["param_n_features_to_select"][j],
                    'scoring': self.settings['resampling']['scoring'],
                    'mean_test_score': grid_result.cv_results_['mean_test_score'][j],
                    'std_test_score': grid_result.cv_results_['std_test_score'][j],
                })

            best_rows.append({
                'model': gs.estimator,
                'cv': cv[i].__str__().split('(')[0],
                'n_splits': int(cv[i].get_n_splits(X) / cv[i].n_repeats),
                'n_repeats': cv[i].n_repeats,
                'random_state': cv[i].random_state,
                'Y': ','.join(Y_cols),
                'n_features_Y': len(Y_cols),
                'n_features_X': grid_result.cv_results_["param_n_features_to_select"][grid_result.best_index_],
                'scoring': self.settings['resampling']['scoring'],
                'mean_test_score': grid_result.cv_results_['mean_test_score'][grid_result.best_index_],
                'std_test_score': grid_result.cv_results_['std_test_score'][grid_result.best_index_],
            })

            # plotting cv results
            plt.figure(figsize=(15, 5))
            plt.plot(grid_result.cv_results_["param_n_features_to_select"],
                     grid_result.cv_results_["mean_train_score"])
            plt.plot(grid_result.cv_results_["param_n_features_to_select"],
                     grid_result.cv_results_["mean_test_score"])
            plt.xlabel('Number of features')
            plt.ylabel('R-squared')
            plt.title(
                f"Optimal Number of Features with {cv[i].__str__().split('(')[0]}(n_splits={int(cv[i].get_n_splits(X) / cv[i].n_repeats)}, n_repeats{cv[i].n_repeats})")
            plt.legend(['train score', 'test score'], loc='upper left')
            plt.savefig(os.path.join(os.getcwd(), 'results', 'cross_validation', 'plot',
                                     f"{cv[i].__str__().split('(')[0].lower()}_n_splits_{int(cv[i].get_n_splits(X) / cv[i].n_repeats)}"))

        pd.DataFrame(rows).to_csv(
            os.path.join(os.getcwd(), 'results', 'cross_validation', 'resampling_regression.csv'), index=False,
            header=True, sep='\t', encoding='utf-8')
        print(f'>>> The file "resampling_regression.csv" has been saved')
        print(f'{"-" * 25}')

        pd.DataFrame(best_rows).to_csv(
            os.path.join(os.getcwd(), 'results', 'cross_validation', 'best_resampling_regression.csv'), index=False,
            header=True, sep='\t', encoding='utf-8')
        print(f'>>> The file "best_resampling_regression.csv" has been saved')
        print(f'{"-" * 25}')

    def calculate(self):
        # Loading the best resampling results
        df = pd.read_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'best_resampling_regression.csv'), sep='\t')

        combs = []
        for i, v in df.iterrows():
            Y_cols = v['Y'].split(',')

            X = self.objDS.dataset[self.objDS.dataset.columns.difference(Y_cols)].select_dtypes(include=['int64', 'float64'])

            X_cols = list(combinations(X.columns.values, v['n_features_X']))
            X_cols = list(map(list, X_cols))  # convert tuple to list

            test_size = math.ceil(self.objDS.dataset.shape[0]/v['n_splits'])
            combs.append([[c, Y_cols, test_size] for c in X_cols])

        # flatten list
        combs = list(chain(*combs))
        print(f'Number of tests to be carried out: {len(combs)}')
        r = Parallel(n_jobs=mlp.cpu_count())(delayed(self.linear_regression)(combs[i]) for i in range(0, len(combs), 1))

        pd.DataFrame(r).to_csv(os.path.join(os.getcwd(), 'results', 'regression', 'regression.csv'),
                               index=False, header=True, sep='\t', encoding='utf-8')
        print(f'>>> The file "regression.csv" has been saved')
        print(f'{"-" * 25}')

    def linear_regression(self, items: list):
        X_train, X_test, y_train, y_test = train_test_split(self.objDS.dataset[items[0]], self.objDS.dataset[items[1]],
                                                            test_size=items[2], random_state=self.init_settings['seed'])

        model = LinearRegression(fit_intercept=True, positive=True)
        model = model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        n_obs, n_regressors = self.objDS.dataset[items[0]].shape
        if len(items[1]) == 1 and len(items[0]) == 1:
            type_lm = 'Linear regression'
        elif len(items[1]) == 1 and len(items[0]) > 1:
            type_lm = 'Multiple linear regression'
        elif len(items[1]) > 1 and len(items[0]) > 1:
            type_lm = 'Multivariate linear regression'

        r = {
            'method': type_lm,
            'Y': ','.join(items[1]),
            'X': ','.join(items[0]),
            'train_size': (self.objDS.dataset.shape[0] - items[2]),
            'test_size': items[2],
            'random_state': self.init_settings['seed'],
            'r2_score_test': sm.r2_score(y_test, y_pred),
            'adj_r2_score_test': (1 - (1 - sm.r2_score(y_test, y_pred)) * (n_obs - 1) / (n_obs - n_regressors - 1)),
            'mean_absolute_error': sm.mean_absolute_error(y_test, y_pred),
            'mean_squared_error': sm.mean_squared_error(y_test, y_pred),
            'median_absolute_error': sm.median_absolute_error(y_test, y_pred),
            'explain_variance_score': sm.explained_variance_score(y_test, y_pred)
        }
        return r

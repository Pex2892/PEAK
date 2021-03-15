import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
from itertools import combinations, chain, product
import multiprocessing as mlp
from joblib import Parallel, delayed

class Regression:

    def automate(self, dataset):
        df = pd.read_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'CV_best_features_regression.csv'), sep='\t')

        to_test = []
        for i, v in df.iterrows():
            colsY = v['cols_Y'].split(',')
            colsX = v['cols_X'].split(',')

            # creo le combinazioni se le features sono maggiori di 1
            colsX = list(combinations(colsX, v['n_features_X']))
            colsX = list(map(list, colsX))  # convert tuple to list

            to_test.append([[c, colsY, v['train_size'], v['test_size']] for c in colsX])

        to_test = list(chain(*to_test))  # flatten list
        r = Parallel(n_jobs=mlp.cpu_count(), verbose=5)(delayed(self.lm)
                                                        (dataset, t[0], t[1], t[2], t[3]) for t in to_test)
        df_lm = pd.DataFrame(r)
        df_lm.to_csv(os.path.join(os.getcwd(), 'results', 'regression', 'regression.csv'),
                             index=False, header=True, sep='\t', encoding='utf-8')

    def lm(self, dataset, X, Y, train_size: float = 0.7, test_size: float = 0.3):
        X_train, X_test, y_train, y_test = train_test_split(dataset[X], dataset[Y], train_size=train_size, test_size=test_size, random_state=None)

        model = LinearRegression(fit_intercept=True, positive=True)
        model = model.fit(X_train, y_train)  # passiamo i set di addestramento

        y_pred_train = model.predict(X_train)  # eseguiamo la predizione sul test set
        y_pred_test = model.predict(X_test)  # eseguiamo la predizione sul test set

        type_lm = None
        r2_score_train = None
        adj_r2_score_train = 'NaN'
        r2_score_test = None
        adj_r2_score_test = 'NaN'
        if len(Y) == 1 and len(X) == 1:
            type_lm = 'Linear regression'
            r2_score_train = sm.r2_score(y_train, y_pred_train)
            r2_score_test = sm.r2_score(y_test, y_pred_test)
        elif len(Y) == 1 and len(X) > 1:
            type_lm = 'Multiple linear regression'
            n_obs, n_regressors = dataset[X].shape
            r2_score_train = sm.r2_score(y_train, y_pred_train)
            adj_r2_score_train = 1 - (1 - r2_score_train) * (n_obs - 1) / (n_obs - n_regressors - 1)
            r2_score_test = sm.r2_score(y_test, y_pred_test)
            adj_r2_score_test = 1 - (1 - r2_score_test) * (n_obs - 1) / (n_obs - n_regressors - 1)
        elif len(Y) > 1 and len(X) > 1:
            type_lm = 'Multivariate linear regression'
            n_obs, n_regressors = dataset[X].shape
            r2_score_train = sm.r2_score(y_train, y_pred_train)
            adj_r2_score_train = 1 - (1 - r2_score_train) * (n_obs - 1) / (n_obs - n_regressors - 1)
            r2_score_test = sm.r2_score(y_test, y_pred_test)
            adj_r2_score_test = 1 - (1 - r2_score_test) * (n_obs - 1) / (n_obs - n_regressors - 1)

        r = {
            'method': type_lm,
            'Y': ','.join(Y),
            'X': ','.join(X),
            'train_size': train_size,
            'test_size': test_size,
            'r2_score_train': r2_score_train,
            'adj_r2_score_train': adj_r2_score_train,
            'r2_score_test': r2_score_test,
            'adj_r2_score_test': adj_r2_score_test,
            'mean_absolute_error': sm.mean_absolute_error(y_test, y_pred_test),
            'mean_squared_error': sm.mean_squared_error(y_test, y_pred_test),
            'median_absolute_error': sm.median_absolute_error(y_test, y_pred_test),
            'explain_variance_score': sm.explained_variance_score(y_test, y_pred_test)
        }
        return r

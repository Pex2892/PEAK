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

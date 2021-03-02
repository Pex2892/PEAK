import os
import random
import pandas as pd
import numpy as np
import multiprocessing as mlp
from itertools import combinations, chain
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class HyperParams:

    def best_split_dataset(self, df, X_cols, Y_cols, n_comb: int = 3, f_name: str = 'best_split_dataset.csv'):
        combs = [list(combinations(X_cols, i)) for i in range(1, len(X_cols)+1)]

        flatten = list(chain(*combs))
        # print(len(flatten), flatten)

        test_size = np.arange(0.3, 0.5, 0.01)
        # print(len(test_size), test_size)

        test_list = []
        for cols in flatten:
            for size in test_size:
                randomlist = random.sample(range(0, 100000), n_comb)
                for seed in randomlist:
                    test_list.append([list(cols), round(size, 2), seed])
        # print(len(test_list), test_list)

        r = Parallel(n_jobs=mlp.cpu_count(), verbose=10)(
            delayed(self.par_linear_regression)
            (df[test[0]], df[Y_cols], test[1], test[2]) for test in test_list)

        df_result = pd.DataFrame(r, columns=['X', 'Y', 'test_size', 'random_state', 'r2_score'])
        df_group = df_result.groupby(['X'])

        r = Parallel(n_jobs=mlp.cpu_count(), verbose=10)(
            delayed(self.par_max_r2score_group)
            (df_group, group) for group in df_group.groups.keys())

        df_result = df_result.iloc[r, :].sort_values(by=['r2_score'], ascending=False)

        df_result.to_csv(os.path.join(os.getcwd(), 'results', f'HyperSplit{"_".join(Y_cols)}.csv'), index=False, header=True, sep='\t', encoding='utf-8')

    def par_linear_regression(self, X, Y, size, seed):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size, random_state=seed, shuffle=True)
        model = LinearRegression(fit_intercept=True, normalize=True, copy_X=False, n_jobs=-1, positive=True)
        model = model.fit(X_train, Y_train)  # passiamo i set di addestramento

        Y_pred = model.predict(X_test)  # eseguiamo la predizione sul test set

        row = {
            'X': ', '.join(X.columns),
            'Y': ', '.join(Y.columns),
            'test_size': size,
            'random_state': seed,
            'r2_score': r2_score(Y_test, Y_pred)
        }
        return row

    def par_max_r2score_group(self, df, group):
        return df.get_group(group)['r2_score'].idxmax()



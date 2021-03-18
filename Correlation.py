import os
import re
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mlp
from joblib import Parallel, delayed
from itertools import combinations
from matplotlib import colors


class Correlation:

    def automate(self, ds):
        cols = ds.dataset.select_dtypes(include=['int64', 'float64']).columns.values
        cc = list(combinations(cols, 2))

        df_corr = pd.DataFrame(columns=['columns', 'pearson', 'spearman', 'kendall'])

        r = Parallel(n_jobs=mlp.cpu_count())(delayed(self.corr_2_col)(ds.dataset, list(c), None, False, (8, 5)) for c in cc)
        df_corr = df_corr.append(r, ignore_index=True)
        print('>>> Plotted all correlations between numeric columns')

        self.matrix_corr(ds.dataset, ['pearson', 'spearman', 'kendall'], show=False, figsize=(15, 7))

        df_corr.to_csv(os.path.join(os.getcwd(), 'results', 'correlation', 'all_correlations.csv'), index=False, header=True, sep='\t', encoding='utf-8')

    def corr_2_col(self, df, cols, title: str = None, show: bool = True, figsize: tuple = (10, 5)):
        scores = [df[cols].corr(method='pearson').iloc[0, 1], df[cols].corr(method='spearman').iloc[0, 1],
                  df[cols].corr(method='kendall').iloc[0, 1]]

        if not np.isnan(scores).any() and len(set(scores)) != 1:
            plt.figure(figsize=figsize)
            cmap = plt.cm.get_cmap('GnBu', 4)
            c_list = [colors.rgb2hex(cmap(i)) for i in range(1, cmap.N)]

            plt.bar(['Pearson', 'Spearman', 'Kendall'], scores, color=c_list)

            if title is None:
                plt.title(f'Corr: {cols[0]} and {cols[1]}')
            else:
                plt.title(title)

            if show:
                plt.show()
            else:
                plt.savefig(os.path.join(os.getcwd(), 'results', 'correlation', 'plot', f'corr_{"".join(re.split("[^a-zA-Z]*", cols[0]))}_{"".join(re.split("[^a-zA-Z]*", cols[1]))}.png'))

        row = {'columns': ' - '.join(cols), 'pearson': scores[0], 'spearman': scores[1], 'kendall': scores[2]}

        return row

    def matrix_corr(self, df, methods: list=['pearson'], show: bool = True, figsize=(10, 5)):
        for method in methods:
            corrMatrix = df.corr(method=method)
            plt.figure(figsize=figsize)
            sns.heatmap(corrMatrix, cmap='GnBu', linewidths=1.0, annot=True)
            plt.title(f'Correlation matrix with {method.capitalize()} correlation coeff.')
            plt.tight_layout()
            if show:
                plt.show()
            else:
                plt.savefig(os.path.join(os.getcwd(), 'results', 'correlation', 'matrix', f'corr_matrix_{method}.png'))
            print(f'>>> Plotted correlation matrix using {method.capitalize()}')

        plt.close()

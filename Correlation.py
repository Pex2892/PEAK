import os
import seaborn as sns
import matplotlib.pyplot as plt


class Correlation:
    def matrix_corr(self, df, methods: list=['pearson'], show=True, figsize=(10, 5)):
        for method in methods:
            corrMatrix = df.corr(method=method)
            plt.figure(figsize=figsize)
            sns.heatmap(corrMatrix, cmap='GnBu', linewidths=1.0, annot=True)
            plt.title(f'Correlation matrix with {method.capitalize()} correlation coeff.')
            plt.tight_layout()
            if show:
                plt.show()
            else:
                plt.savefig(os.path.join(os.getcwd(), 'results', f'corrMatrix_{method}.png'))

        plt.close()
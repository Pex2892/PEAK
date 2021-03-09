import os
import re
import numpy as np
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt


class Explore:

    def automate(self, obj):
        cols = obj.dataset.select_dtypes(include=['object']).columns.values

        for c in cols:
            n = pd.unique(obj.dataset[c]).shape[0]
            title = f'Distribution of {"".join(re.split("[^a-zA-Z]*", c))}'
            if 1 < n < 7:
                self.multiple_barplot(data=[[obj.dataset[c], title, 'Labels', 'Frequency']], tot_subplots=1, tot_columns=1, title='', show=True)
            else:
                self.multiple_barplot_horizontal(data=[[obj.dataset[c], title, 'Frequency', 'Labels', 1]], tot_subplots=1, tot_columns=1, title='', show=True)


    def multiple_barplot(self, data: list, tot_subplots: int, tot_columns: int, title: str, show=True, figname='multiplebarplot.png', figsize=(10, 5)):
        # Compute Rows required
        rows = tot_subplots // tot_columns
        rows += tot_subplots % tot_columns

        # Create a Position index
        position = range(1, tot_subplots + 1)

        # Create main figure
        fig = plt.figure(1, figsize=figsize)

        if tot_subplots > 1:
            fig.suptitle(f'{title}{data[0][0].shape[0]} Samples')
        else:
            fig.suptitle(f'{data[0][1]} ({data[0][0].shape[0]} Samples)')

        for k in range(tot_subplots):
            # add every single subplot to the figure with a for loop
            ax = fig.add_subplot(rows, tot_columns, position[k])

            height = data[k][0].value_counts().to_numpy()
            bars = data[k][0].unique()

            cmap = plt.cm.get_cmap('tab20', len(bars))
            c = [colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

            y_pos = np.arange(len(bars))
            ax.bar(y_pos, height, color=c)
            ax.set_xticks(y_pos)
            ax.set_xticklabels(bars)
            if tot_subplots > 1:
                ax.set_title(data[k][1])
            ax.set_ylabel(data[k][3])
            ax.set_xlabel(data[k][2])
            ax.grid(axis='y', linestyle='--')

        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(os.getcwd(), 'results', figname))
        plt.close()

    def multiple_barplot_horizontal(self, data: list, tot_subplots: int, tot_columns: int, title: str, show=True, figname='multiplebarplot.png', figsize=(10, 5)):
        # data = [numpy_array, title subplot, xlabel, ylabel, xticks_step]

        # Compute Rows required
        rows = tot_subplots // tot_columns
        rows += tot_subplots % tot_columns

        # Create a Position index
        position = range(1, tot_subplots + 1)

        # Create main figure
        fig = plt.figure(1, figsize=figsize)
        if tot_subplots > 1:
            fig.suptitle(f'{title}{data[0][0].shape[0]} Samples')
        else:
            fig.suptitle(f'{data[0][1]} ({data[0][0].shape[0]} Samples)')

        for k in range(tot_subplots):
            # add every single subplot to the figure with a for loop
            ax = fig.add_subplot(rows, tot_columns, position[k])

            height = data[k][0].value_counts().to_numpy()
            bars = data[k][0].unique()

            cmap = plt.cm.get_cmap('tab20c', len(bars))
            c = [colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

            y_pos = np.arange(len(bars))
            ax.barh(y_pos, height, color=c)
            ax.set_xticks(np.arange(0, height.max()+1, data[k][4]))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(bars)
            if tot_subplots > 1:
                ax.set_title(data[k][1])
            ax.set_ylabel(data[k][3])
            ax.set_xlabel(data[k][2])

        plt.margins(y=0.01)
        plt.grid(axis='x', linestyle='--')
        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(os.getcwd(), 'results', figname))
        plt.close()

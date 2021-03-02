import os
import numpy as np
import matplotlib.pyplot as plt


class Explore:
    def multiple_barplot(self, data: list, tot_subplots: int, tot_columns: int, title: str, show=True, figname='multiplebarplot.png', figsize=(10, 5)):
        # Compute Rows required
        Rows = tot_subplots // tot_columns
        Rows += tot_subplots % tot_columns

        # Create a Position index
        Position = range(1, tot_subplots + 1)

        # Create main figure
        fig = plt.figure(1, figsize=figsize)
        fig.suptitle(f'{title}{data[0][0].shape[0]} Samples')
        for k in range(tot_subplots):
            # add every single subplot to the figure with a for loop
            ax = fig.add_subplot(Rows, tot_columns, Position[k])

            height = data[k][0].value_counts().to_numpy()
            bars = data[k][0].unique()
            y_pos = np.arange(len(bars))
            ax.bar(y_pos, height)
            ax.set_xticks(y_pos)
            ax.set_xticklabels(bars)
            ax.set_title(data[k][1])
            ax.set_ylabel(data[k][3])
            ax.set_xlabel(data[k][2])
            ax.grid(axis='y', linestyle='--')

        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(os.getcwd(), 'results', figname))

    def multiple_barplot_horizontal(self, data: list, tot_subplots: int, tot_columns: int, title: str, show=True, figname='multiplebarplot.png', figsize=(10, 5)):
        # data = [numpy_array, title subplot, xlabel, ylabel, xticks_step]

        # Compute Rows required
        Rows = tot_subplots // tot_columns
        Rows += tot_subplots % tot_columns

        # Create a Position index
        Position = range(1, tot_subplots + 1)

        # Create main figure
        fig = plt.figure(1, figsize=figsize)
        fig.suptitle(f'{title}{data[0][0].shape[0]} Samples')
        for k in range(tot_subplots):
            # add every single subplot to the figure with a for loop
            ax = fig.add_subplot(Rows, tot_columns, Position[k])

            height = data[k][0].value_counts().to_numpy()
            bars = data[k][0].unique()
            y_pos = np.arange(len(bars))
            ax.barh(y_pos, height)
            ax.set_xticks(np.arange(0, height.max()+1, data[k][4]))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(bars)
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
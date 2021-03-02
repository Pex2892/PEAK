import os
import pandas as pd
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Classifier:
    def __init__(self, X, Y, labels, tt_set, test_size, seed):
        self.df = pd.DataFrame(columns=['classifier', 'X', 'Y', 'dim_test_set', 'random_state', 'mc_true_negative',
                                        'mc_false_positive', 'mc_false_negative', 'mc_true_positive', 'accuracy',
                                        'precision', 'recall', 'f1', 'roc_curve_auc'])

        self.X = X
        self.Y_numeric = Y
        self.Y_categorial = labels
        self.tt_set = tt_set
        self.test_size = test_size
        self.seed = seed
        self.all_roc_curve = []

    def calculate(self, model, name_model: str, show_plot: bool, fig_name: str):
        model = model.fit(self.tt_set[0], self.tt_set[2])  # passiamo i set di addestramento
        probas_ = model.predict_proba(self.tt_set[1])

        # Predicting the Test set results
        Y_pred = model.predict(self.tt_set[1])  # eseguiamo la predizione sul test set

        # calcolo ed estraggo la matrice di confusione
        # confusion_matrix è solo per 2 label, se di più si usa multilabel_confusion_matrix
        tn, fp, fn, tp = sm.confusion_matrix(self.tt_set[3], Y_pred).ravel()

        # Posso sapere tutte le caratteristiche del classificatore
        # print(sm.classification_report(Y_test, Y_pred))

        fpr, tpr, thresholds = sm.roc_curve(self.tt_set[3], probas_[:, 1])

        scores = [
            sm.accuracy_score(self.tt_set[3], Y_pred),
            sm.precision_score(self.tt_set[3], Y_pred),
            sm.recall_score(self.tt_set[3], Y_pred),
            sm.f1_score(self.tt_set[3], Y_pred),
            sm.auc(fpr, tpr)
        ]

        row = {
            'classifier': name_model,
            'X': ', '.join(self.X.columns.values),
            'Y': ', '.join(self.Y_numeric.columns.values),
            'dim_test_set': self.test_size,
            'random_state': self.seed,
            'mc_true_negative': tn,
            'mc_false_positive': fp,
            'mc_false_negative': fn,
            'mc_true_positive': tp,
            'accuracy': scores[0],
            'precision': scores[1],
            'recall': scores[2],
            'f1': scores[3],
            'roc_curve_auc': scores[4]
        }
        self.df = self.df.append(row, ignore_index=True)

        self.all_roc_curve.append([name_model, fpr, tpr, round(scores[4], 2)])

        self.plot_classifier(model, fpr, tpr, scores, show_plot,
                             f'{name_model} – {",".join(self.Y_numeric.columns.values)} ~ {",".join(self.X.columns.values)}',
                             fig_name)

    def plot_classifier(self, model, fpr, tpr, scores, show_plot, title_plot, figname: str):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        fig.suptitle(f'{title_plot}')

        # 1° Plot – Matrice di confusione
        sm.plot_confusion_matrix(model, self.tt_set[1], self.tt_set[3], display_labels=self.Y_categorial, cmap='GnBu',
                                 normalize=None, ax=axs[0])
        axs[0].set_title(f'Matrix Confusion')

        # 2° Plot – Misure di performance
        height = scores[0:-1]
        bars = ['Accuracy', 'Precision', 'Recall', 'F1']
        colors = ['#0868ac', '#2b8cbe', '#7bccc4', '#bae4bc']
        handles = [mpatches.Patch(color=colors[i], label=round(scores[i], 2)) for i in range(0, len(bars))]
        axs[1].bar(bars, height, color=colors)
        axs[1].set_xlabel('Metrics')
        axs[1].set_ylabel('Score')
        axs[1].set_title(f'Metrics and Scoring')
        axs[1].grid(axis='y', linestyle='--')
        axs[1].legend(handles=handles)
        axs[1].set_ylim(0.0, 1.0)

        # 3° Plot – Curva ROC
        axs[2].plot(fpr, tpr, color=colors[0], label=f'AUC = {round(scores[4], 2)}')
        axs[2].plot([0, 1], [0, 1], color='#DCDCDC', linestyle='--')
        axs[2].set_xlim([0.0, 1.0])
        axs[2].set_ylim([0.0, 1.0])
        axs[2].set_xlabel('False Positive Rate')
        axs[2].set_ylabel('True Positive Rate')
        axs[2].legend(loc="lower right")
        axs[2].set_title(f'ROC Curve')

        fig.tight_layout()

        if show_plot:
            plt.show()
        else:
            plt.savefig(os.path.join(os.getcwd(), 'results', figname))
        plt.close()

    def plot_all_roc_curve(self, show_plot, title_plot, fig_name: str):
        plt.figure(figsize=(8, 5))
        for item in self.all_roc_curve:
            plt.plot(item[1], item[2], label=f'{item[0]} - AUC = {item[3]}')
        plt.plot([0, 1], [0, 1], color='#DCDCDC', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title(title_plot)
        plt.tight_layout()

        if show_plot:
            plt.show()
        else:
            plt.savefig(os.path.join(os.getcwd(), 'results', fig_name))

    def export_csv(self, f: str):
        self.df.to_csv(os.path.join(os.getcwd(), 'results', f), index=False, header=True, sep='\t', encoding='utf-8')



import os
import pandas as pd
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import ast
from sklearn.model_selection import train_test_split
import math


class Classifier:

    def automate(self, dataset):
        # Loading the best results obtained from hyper parameters
        df = pd.read_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'CV_best_hyperparams_classification.csv'), sep='\t')

        roc_curve_clf = []
        for i, v in df.iterrows():
            model = self._recognize_clf(v['model'], v['parameters'], v['random_state'])
            model_name = v['model'].replace('()', '')
            colsY = v['Y'].split(',')
            colsX = v['X'].split(',')
            test_size = math.ceil(dataset[colsX].shape[0] / v['n_splits'])

            r = self.calculate(dataset, model, model_name, colsX, colsY, test_size=test_size,
                               seed=v['random_state'], show_plot=False)
            roc_curve_clf.append(r)

        # plot all roc curve
        self.plot_all_roc_curve(roc_curve_clf, show_plot=False)

    def _recognize_clf(self, m, p, s):
        params = ast.literal_eval(p)
        if m == 'LogisticRegression()':
            model = LogisticRegression()
            model.C = params['C']
            model.penalty = params['penalty']
            model.solver = params['solver']
        elif m == 'KNeighborsClassifier()':
            model = KNeighborsClassifier()
            model.n_neighbors = params['n_neighbors']
            model.weights = params['weights']
            model.metric = params['metric']
            model.algorithm = params['algorithm']
        elif m == 'SVC()':
            model = SVC()
            model.C = params['C']
            model.gamma = params['gamma']
            model.kernel = params['kernel']
            model.probability = True
        elif m == 'RandomForestClassifier()':
            model = RandomForestClassifier()
            model.class_weight = params['class_weight']
            model.criterion = params['criterion']
            model.max_features = params['max_features']
            model.n_estimators = params['n_estimators']
        elif m == 'GaussianNB()':
            model = GaussianNB()
            model.var_smoothing = params['var_smoothing']
        elif m == 'DecisionTreeClassifier()':
            model = DecisionTreeClassifier()
            model.class_weight = params['class_weight']
            model.criterion = params['criterion']
            model.max_features = params['max_features']
            model.splitter = params['splitter']
        elif m == 'MLPClassifier()':
            model = MLPClassifier()
            model.activation = params['activation']
            model.alpha = params['alpha']
            model.hidden_layer_sizes = params['hidden_layer_sizes']
            model.learning_rate = params['learning_rate']
            model.max_iter = params['max_iter']
            model.solver = params['solver']
        else:
            print('ERROR')
            exit()

        model.random_state = s
        return model

    def calculate(self, dataset, model, model_name: str, colsX, colsY, test_size, seed, show_plot: bool = True, fig_name: str = None):

        X_train, X_test, Y_train, Y_test = train_test_split(dataset[colsX], dataset[colsY], test_size=test_size, random_state=None)

        m = model.fit(X_train, Y_train)
        probas_ = m.predict_proba(X_test)

        # Predicting the Test set results
        Y_pred = model.predict(X_test)

        # the confusion matrix
        # the "confusion_matrix" method is used only with 2 categories
        # if more than 2 you must use the "multilabel_confusion_matrix" method
        tn, fp, fn, tp = sm.confusion_matrix(Y_test, Y_pred).ravel()

        # Build a text report showing the main classification metrics.
        # print(sm.classification_report(Y_test, Y_pred))

        # ROC CURVE
        fpr, tpr, thresholds = sm.roc_curve(Y_test, probas_[:, 1])

        # This list contains the following metrics: accuracy, precision, recall, F1, AUC
        scores = [
            sm.accuracy_score(Y_test, Y_pred),
            sm.precision_score(Y_test, Y_pred),
            sm.recall_score(Y_test, Y_pred),
            sm.f1_score(Y_test, Y_pred),
            sm.auc(fpr, tpr)
        ]

        # PLOT
        if fig_name is None:
            fig_name = f'{model_name}_clf.png'

        p = {
            'model': m,
            'X_test': X_test,
            'Y_test': Y_test,
            'Y_label': dataset.iloc[:, dataset.columns.get_loc(colsY[0])-1].unique(),
            'fpr': fpr,
            'tpr': tpr,
            'scores': scores
        }

        self.plot_classifier(p, show_plot, f'{model_name} – {",".join(colsY)} ~ {",".join(colsX)}', fig_name)

        # export
        row = {
            'classifier': model_name,
            'X': ', '.join(colsX),
            'Y': ', '.join(colsY),
            'dim_test_set': test_size,
            'random_state': seed,
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
        p = os.path.join(os.getcwd(), 'results', 'classification', 'classification.csv')
        pd.DataFrame([row]).to_csv(p, mode='a', index=False, header=not os.path.exists(p), sep='\t', encoding='utf-8')

        return [model_name, fpr, tpr, round(scores[4], 2)]

    def plot_classifier(self, p: dict, show_plot: bool, title_plot: str, figname: str):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        fig.suptitle(f'{title_plot}')

        # 1° Plot – Matrice di confusione
        sm.plot_confusion_matrix(p['model'], p['X_test'], p['Y_test'], display_labels=p['Y_label'], cmap='GnBu',
                                 normalize=None, ax=axs[0])
        axs[0].set_title(f'Matrix Confusion')

        # 2° Plot – Misure di performance
        height = p['scores'][0:-1]
        bars = ['Accuracy', 'Precision', 'Recall', 'F1']
        colors = ['#0868ac', '#2b8cbe', '#7bccc4', '#bae4bc']
        handles = [mpatches.Patch(color=colors[i], label=round(p['scores'][i], 2)) for i in range(0, len(bars))]
        axs[1].bar(bars, height, color=colors)
        axs[1].set_xlabel('Metrics')
        axs[1].set_ylabel('Score')
        axs[1].set_title(f'Metrics and Scoring')
        axs[1].grid(axis='y', linestyle='--')
        axs[1].legend(handles=handles)
        axs[1].set_ylim(0.0, 1.0)

        # 3° Plot – Curva ROC
        axs[2].plot(p['fpr'], p['tpr'], color=colors[0], label=f"AUC = {round(p['scores'][4], 2)}")
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
            plt.savefig(os.path.join(os.getcwd(), 'results', 'classification', 'plot', figname))
        plt.close()

    def plot_all_roc_curve(self, roc_curve, show_plot: bool):
        plt.figure(figsize=(8, 5))
        for item in roc_curve:
            plt.plot(item[1], item[2], label=f'{item[0]} - AUC = {item[3]}')
        plt.plot([0, 1], [0, 1], color='#DCDCDC', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title('ROC curve of all classifiers')
        plt.tight_layout()

        if show_plot:
            plt.show()
        else:
            plt.savefig(os.path.join(os.getcwd(), 'results', 'classification', 'plot', 'all_roc_curve.png'))

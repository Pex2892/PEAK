import os
import ast
import numpy as np
import pandas as pd
import math
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import sys

# The only way I could suppress all Scikit-learn warnings,
# is by issuing the following code at the beginning of the module
# (but note that will suppress all warnings).
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


class Classifier:
    def __init__(self, ds, settings, init_settings):
        self.objDS = ds
        self.settings = settings
        self.init_settings = init_settings

    def automate(self):
        if self.settings['enable'] == 1:
            self._resampling()
            print('>>> Resampling with different classifiers has been completed.')

            self._calculate()

    def _resampling(self):
        # example of grid searching key hyperparametres for logistic regression
        # define dataset

        Y_col = f"{self.settings['y']}_fact"
        print(f'Columns in Y: {Y_col}')

        X = self.objDS.dataset[self.objDS.dataset.columns.difference([self.settings['y'], f"{self.settings['y']}_fact"])].select_dtypes(include=['int64', 'float64'])
        print(f'Columns in X: {X.columns.values}')

        # list of classifiers with different combinations of parameters
        models = [
            [LogisticRegression(), dict(solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], penalty=['l1', 'l2', 'elasticnet', 'none'], C=[100, 10, 1.0, 0.1, 0.01])],
            [KNeighborsClassifier(), dict(n_neighbors=range(1, 21, 2), weights=['uniform', 'distance'], algorithm=['ball_tree', 'kd_tree', 'brute'], metric=['euclidean', 'manhattan', 'minkowski'])],
            [SVC(), dict(kernel=['linear', 'poly', 'rbf', 'sigmoid'], C=[50, 10, 1.0, 0.1, 0.01], gamma=['scale'])],
            [RandomForestClassifier(), dict(n_estimators=[10, 100, 1000], criterion=['gini', 'entropy'], max_features=['sqrt', 'log2', 'none'], class_weight=['balanced', 'balanced_subsample', 'none'])],
            [GaussianNB(), dict(var_smoothing=np.logspace(0, -9, num=100))],
            [DecisionTreeClassifier(), dict(criterion=['gini', 'entropy'], splitter=['best', 'random'], max_features=['sqrt', 'log2', 'none'], class_weight=['balanced', 'none'])],
            [MLPClassifier(), dict(hidden_layer_sizes=[(50, 50, 50), (50, 100, 50), (100,)], activation=['identity', 'logistic', 'tanh', 'relu'], solver=['lbfgs', 'sgd', 'adam'], alpha=[0.0001, 0.05], learning_rate=['constant', 'invscaling', 'adaptive'], max_iter=[100, 200])]
        ]
        cv = [RepeatedStratifiedKFold(n_splits=i, n_repeats=self.settings['resampling']['n_repeats'], random_state=self.init_settings['seed'])
              for i in range(self.settings['resampling']['min_split'], self.settings['resampling']['max_split'] + 1, 1)]

        combs = list(product(*[models, cv]))
        print(f'>>> Number of tests to be carried out: {len(combs)}')

        rows = list()
        best_rows = list()
        for i in range(0, len(combs)):
            print(f'{i+1}) Combination being tested: {combs[i]}')

            # 3.2 call GridSearchCV()
            gs = GridSearchCV(estimator=combs[i][0][0], param_grid=combs[i][0][1], n_jobs=self.init_settings['cpu'], cv=combs[i][1], scoring=self.settings['resampling']['scoring'], error_score=0)
            grid_result = gs.fit(X, self.objDS.dataset.loc[:, Y_col])

            for j in range(0, grid_result.cv_results_['mean_test_score'].shape[0], 1):
                rows.append({
                    'model': gs.estimator,
                    'X': ','.join(X.columns.values),
                    'Y': Y_col,
                    'cv': combs[i][1].__str__().split('(')[0],
                    'n_splits': int(combs[i][1].get_n_splits(X) / combs[i][1].n_repeats),
                    'n_repeats': combs[i][1].n_repeats,
                    'random_state': combs[i][1].random_state,
                    'scoring': self.settings['resampling']['scoring'],
                    'mean_test_score': grid_result.cv_results_['mean_test_score'][j],
                    'std_test_score': grid_result.cv_results_['std_test_score'][j],
                    'parameters': grid_result.cv_results_['params'][j]
                })

            best_rows.append({
                'model': gs.estimator,
                'X': ','.join(X.columns.values),
                'Y': Y_col,
                'cv': combs[i][1].__str__().split('(')[0],
                'n_splits': int(combs[i][1].get_n_splits(X) / combs[i][1].n_repeats),
                'n_repeats': combs[i][1].n_repeats,
                'random_state': combs[i][1].random_state,
                'scoring': self.settings['resampling']['scoring'],
                'mean_test_score': grid_result.cv_results_['mean_test_score'][grid_result.best_index_],
                'std_test_score': grid_result.cv_results_['std_test_score'][grid_result.best_index_],
                'parameters': grid_result.cv_results_['params'][grid_result.best_index_]
            })

        pd.DataFrame(rows).to_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'resampling_classification.csv'),
                                  index=False, header=True, sep='\t', encoding='utf-8')
        print(f'>>> The file "resampling_classification.csv" has been saved')
        print(f'{"-" * 25}')

        # Extracting for each classifier the best combination of parameters with the smallest standard deviation
        pd.DataFrame(best_rows).to_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'best_resampling_classification.csv'),
                                       index=False, header=True, sep='\t', encoding='utf-8')
        df_best = pd.read_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'best_resampling_classification.csv'), sep='\t')
        df_group = df_best.groupby(by=['model'])
        idx = df_group['mean_test_score'].idxmin().to_list()
        df_best.iloc[idx, :].to_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'best_resampling_classification.csv'),
                                    index=False, header=True, sep='\t', encoding='utf-8')
        print(f'>>> The file "best_resampling_classification.csv" has been saved')
        print(f'{"-" * 25}')

    def _calculate(self):
        # Loading the best results obtained from hyper parameters
        df = pd.read_csv(os.path.join(os.getcwd(), 'results', 'cross_validation', 'best_resampling_classification.csv'), sep='\t')

        roc_curve_clf = []
        for i, v in df.iterrows():
            model = self._recognize_clf(v['model'], v['parameters'], v['random_state'])
            model_name = v['model'].replace('()', '')
            Y_col = v['Y']
            X_cols = v['X'].split(',')
            test_size = math.ceil(self.objDS.dataset.shape[0] / v['n_splits'])

            r = self._classifier(model, model_name, X_cols, Y_col, test_size=test_size, seed=v['random_state'])
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

    def _classifier(self, model, model_name: str, X_cols, Y_col, test_size, seed):
        X_train, X_test, y_train, y_test = train_test_split(self.objDS.dataset[X_cols], self.objDS.dataset[Y_col],
                                                            test_size=test_size, random_state=seed)

        m = model.fit(X_train, y_train)
        probas_ = m.predict_proba(X_test)

        # Predicting the Test set results
        y_pred = model.predict(X_test)

        # the confusion matrix
        # the "confusion_matrix" method is used only with 2 categories
        # if more than 2 you must use the "multilabel_confusion_matrix" method
        tn, fp, fn, tp = sm.confusion_matrix(y_test, y_pred).ravel()

        # Build a text report showing the main classification metrics.
        # print(sm.classification_report(Y_test, Y_pred))

        # ROC CURVE
        fpr, tpr, thresholds = sm.roc_curve(y_test, probas_[:, 1])

        # This list contains the following metrics: accuracy, precision, recall, F1, AUC
        scores = [
            sm.accuracy_score(y_test, y_pred),
            sm.precision_score(y_test, y_pred),
            sm.recall_score(y_test, y_pred),
            sm.f1_score(y_test, y_pred),
            sm.auc(fpr, tpr)
        ]

        print(Y_col)
        # PLOT
        p = {
            'model': m,
            'X_test': X_test,
            'Y_test': y_test,
            'Y_label': self.objDS.dataset.iloc[:, self.objDS.dataset.columns.get_loc(Y_col) - 1].unique(),
            'fpr': fpr,
            'tpr': tpr,
            'scores': scores
        }

        self.plot_classifier(p, False, f'{model_name} – {Y_col} ~ {",".join(X_cols)}', f'{model_name}_clf.png')

        # export
        row = {
            'classifier': model_name,
            'X': ', '.join(X_cols),
            'Y': Y_col,
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

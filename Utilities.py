import os
import shutil
import argparse
from pathlib import Path
import multiprocessing as mlp
import configparser
import ast

def header():
    t = '========================================================\n' \
         '=          PETAL – Pattern rEcognition frAmewoRk       =\n' \
         '=                        v1.0                          =\n' \
         '=          Last update:     2021/03/10                 =\n' \
         '========================================================\n' \
         '=          E-mail: giuseppe.sgroi@unict.it             =\n' \
         '========================================================\n' \
         '=          PEAK is licensed under CC BY-NC-SA 4.0      =\n' \
         '========================================================'
    print(t)


def read_args():
    cfg = configparser.ConfigParser()
    cfg.read('settings.cfg')

    settings_args = dict(clear=cfg.getint('settings', 'clear'), seed=cfg.getint('settings', 'seed'), cpu=cfg.getint('settings', 'cpu'))

    if settings_args['cpu'] == 0 or settings_args['cpu'] > mlp.cpu_count():
        settings_args['cpu'] = mlp.cpu_count()

    dataset_args = dict(filename=cfg.get('dataset', 'fname'), sep=cfg.get('dataset', 'separator'))

    regression_args = dict(y=cfg.get('regression', 'y'), resampling=ast.literal_eval(cfg.get('regression', 'resampling')))

    classification_args = dict(y=cfg.get('classification', 'y'), resampling=ast.literal_eval(cfg.get('classification', 'resampling')))

    args = dict(settings=settings_args, dataset=dataset_args, regression=regression_args, classification=classification_args)
    print(f'>>> Parameters: {args}')

    return args


def check_args(dataset, Y_regr, Y_clf):
    cols_Y_regr = Y_regr.split(',')

    for c in cols_Y_regr:
        if not c in dataset.columns:
            print(f'La colonna "{c}" non esiste all\'interno del dataset')
            exit()

        if not dataset[c].dtype in ['int64', 'float64']:
            print(f'La colonna "{c}" non è di tipo numerico')
            exit()

    cols_Y_clf = Y_clf.split(',', 1)
    for c in cols_Y_clf:
        if not c in dataset.columns:
            print(f'La colonna "{c}" non esiste all\'interno del dataset')
            exit()

        if not dataset[c].dtype in ['object']:
            print(f'La colonna "{c}" non è di tipo categoriale')


def clear_data(clear: bool):

    if clear == 1:
        p = os.path.join(os.getcwd(), 'results')
        if os.path.exists(p):
            shutil.rmtree(p)

        folders = [
            os.path.join(p, 'exploration'),
            os.path.join(p, 'correlation', 'plot'),
            os.path.join(p, 'correlation', 'matrix'),
            os.path.join(p, 'cross_validation'),
            os.path.join(p, 'regression', 'plot'),
            os.path.join(p, 'regression'),
            os.path.join(p, 'classification'),
            os.path.join(p, 'classification', 'plot'),
        ]

        for p in folders:
            Path(p).mkdir(parents=True, exist_ok=True)

        print('>>> Previous data has been deleted')
    else:
        print('>>> Previous data has not been deleted')

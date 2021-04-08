import os
import shutil
import configparser
import ast
import multiprocessing as mlp
from pathlib import Path


def header():
    t = '========================================================\n' \
         '=          PEAK – Pattern rEcognition frAmewoRk        =\n' \
         '=                        v1.0                          =\n' \
         '=          Last update:     2021/04/01                 =\n' \
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

    dataset_args = dict(filename=cfg.get('dataset', 'fname'), sep=cfg.get('dataset', 'separator'), skiprows=cfg.getint('dataset', 'skiprows'))

    regression_args = dict(enable=cfg.getint('regression', 'enable'), y=cfg.get('regression', 'y'), resampling=ast.literal_eval(cfg.get('regression', 'resampling')))

    classification_args = dict(enable=cfg.getint('classification', 'enable'), y=cfg.get('classification', 'y'), resampling=ast.literal_eval(cfg.get('classification', 'resampling')))

    args = dict(settings=settings_args, dataset=dataset_args, regression=regression_args, classification=classification_args)
    print(f'> Parameters: {args}')

    return args


def clear_data(clear: bool):
    p = os.path.join(os.getcwd(), 'results')
    if clear == 1 or not os.path.exists(p):
        p = os.path.join(os.getcwd(), 'results')
        if os.path.exists(p):
            shutil.rmtree(p)

        folders = [
            os.path.join(p, 'eda'),
            os.path.join(p, 'correlation', 'plot'),
            os.path.join(p, 'correlation', 'matrix'),
            os.path.join(p, 'cross_validation'),
            os.path.join(p, 'cross_validation', 'plot'),
            os.path.join(p, 'regression'),
            os.path.join(p, 'classification'),
            os.path.join(p, 'classification', 'plot'),
        ]

        for p in folders:
            Path(p).mkdir(parents=True, exist_ok=True)

        print('>>> Previous results has been deleted')
    else:
        print('>>> Previous results has not been deleted')

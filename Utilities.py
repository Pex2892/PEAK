import os
import shutil
import argparse
from pathlib import Path
import multiprocessing as mlp


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
    parser = argparse.ArgumentParser(description='PEAK')

    parser.add_argument('-db', '--dataset', type=str, help='Inserire il dataset', default='', required=True)
    parser.add_argument('-yr', '--Y_regr', type=str, help='Colonne dipententi per la regressione', default='')
    parser.add_argument('-yc', '--Y_clf', type=str, help='Colonna dipententi per la classificazione', default='')
    parser.add_argument('-c', '--cpu', type=int, help='(optional) Maximum number of CPUs used '
                                                        'during the analysis – Default value = 0', choices=range(0, mlp.cpu_count()), default=0)

    args = parser.parse_args()

    if args.cpu == 0:
        args.cpu = mlp.cpu_count()

    print(args)

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

def clear_data():
    p = os.path.join(os.getcwd(), 'results')
    if os.path.exists(p):
        shutil.rmtree(p)

    folders = [
        os.path.join(p, 'exploration'),
        os.path.join(p, 'correlation', 'plot'),
        os.path.join(p, 'correlation', 'matrix'),
        os.path.join(p, 'cross_validation'),
        os.path.join(p, 'regression', 'plot'),
        os.path.join(p, 'regression', 'hyper_params'),
        os.path.join(p, 'classifier', 'plot'),
        os.path.join(p, 'classifier', 'hyper_params'),
    ]

    for p in folders:
        Path(p).mkdir(parents=True, exist_ok=True)


# def check_params(params):
#
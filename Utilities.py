import os
import shutil
from pathlib import Path


def clear_data():
    p = os.path.join(os.getcwd(), 'results')
    if os.path.exists(p):
        shutil.rmtree(p)

    folders = [
        os.path.join(p, 'exploration'),
        os.path.join(p, 'correlation', 'plot'),
        os.path.join(p, 'correlation', 'matrix'),
        os.path.join(p, 'kfold'),
        os.path.join(p, 'regression', 'plot'),
        os.path.join(p, 'regression', 'hyper_params'),
        os.path.join(p, 'classifier', 'plot'),
        os.path.join(p, 'classifier', 'hyper_params'),
    ]

    for p in folders:
        Path(p).mkdir(parents=True, exist_ok=True)

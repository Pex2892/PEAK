import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Dataset:
    def __init__(self, settings):
        self.dataset = None
        self.filename = settings['filename']
        self.type_db = os.path.splitext(os.path.join(os.getcwd(), 'dataset', self.filename))[1]
        self.sep = settings['sep']
        self.skiprows = settings['skiprows']

        self.load()
        print(f'>>> The dataset "{self.filename}" has been successfully loaded')

    def load(self):
        if os.path.exists(os.path.join(os.getcwd(), 'dataset', self.filename)):
            if self.type_db == '.csv':
                self.dataset = pd.read_csv(os.path.join(os.getcwd(), 'dataset', self.filename), sep=self.sep)
            elif self.type_db == '.xlsx':
                self.dataset = pd.read_excel(os.path.join(os.getcwd(), 'dataset', self.filename))
        else:
            print(f'The file "{self.filename}" was not found.')
            exit()

    def drop_cols(self, cols: list = None):
        if cols is None:

            # Calculates the frequency of NaN and divides it with respect to the total number of samples
            s_nan = self.dataset.isna().sum() / self.dataset.shape[0]

            # Takes columns that have a NaN frequency greater than 50% (0.5)
            cols_to_del = [i for i, v in s_nan.iteritems() if v > 0.5]

            # Takes columns that have constant value and joins it to the list of columns to delete
            cols_to_del = [*cols_to_del, *self.dataset.columns[self.dataset.nunique() <= 1].values]

            # Drop specified labels from columns.
            self.dataset = self.dataset.drop(cols_to_del, axis=1)

            print(f">>> Removed the following columns: {', '.join(cols_to_del)}")
        else:
            if len(cols) > 0:
                self.dataset = self.dataset.drop(cols, axis=1)
                print(f">>> Removed the following columns: {', '.join(cols)}")
            else:
                print('Error: No column to delete was found')
                exit()

    def drop_rows(self, rows: list = None):
        if rows is None:
            # Drop the rows where at least one element is missing.
            self.dataset = self.dataset.dropna()
            print('>>> Drop the rows where at least one element is missing')
        else:
            if len(rows) > 0:
                self.dataset = self.dataset.drop(rows, axis=0)
                print(f">>> Removed the following rows by index: {rows}")

    def normalize(self, cols: list = None):
        min_max_scaler = MinMaxScaler()

        if cols is None:
            df_dtypes = self.dataset.select_dtypes(include=['int64', 'float64'])
            cols = df_dtypes.columns.values

        if len(cols) > 0:
            self.dataset[cols] = min_max_scaler.fit_transform(self.dataset[cols])
            print(f">>> Columns were normalized between 0 and 1: {', '.join(cols)}")
        else:
            print('Error: No columns to normalize was found')
            exit()

    def categorial_to_numeric(self, cols: list = None, newcols: list = None):
        if cols is None:
            df_object = self.dataset.select_dtypes(include=['object'])
            cols = df_object.columns.values
            newcols = [f'{c}_fact' for c in cols]

        if len(cols) == len(newcols):
            for c, nc in zip(cols, newcols):
                self.dataset.insert(self.dataset.columns.get_loc(c) + 1, nc, pd.factorize(self.dataset[c])[0])
            print(f">>> The following categorical columns were coded into numbers: {', '.join(cols)}")
            print(f">>> The number columns are named as follows: {', '.join(newcols)}")
        else:
            print('The lengths between "cols" and "newcols" are not the same')
            exit()

    def check_cols_regression(self, enable: int, cols: list):
        if enable == 1:
            cols_Y = cols.split(',')

            for c in cols_Y:
                if not c in self.dataset.columns:
                    print(f'La colonna "{c}" non esiste all\'interno del dataset')
                    exit()

                if not self.dataset[c].dtype in ['int64', 'float64']:
                    print(f'La colonna "{c}" non è di tipo numerico')
                    exit()

    def check_cols_classification(self, enable: int, cols: list):
        if enable == 1:
            cols_Y = cols.split(',')

            for c in cols_Y:
                if not c in self.dataset.columns:
                    print(f'La colonna "{c}" non esiste all\'interno del dataset')
                    exit()

                if not self.dataset[c].dtype in ['object']:
                    print(f'La colonna "{c}" non è di tipo categoriale')
                    exit()

    def export_csv(self):
        self.dataset.to_csv(os.path.join(os.getcwd(), 'results', 'dataset_processed.csv'), index=False,
                            header=True, sep='\t', encoding='utf-8')
        print(f">>> A backup of the dataset has been created in the following path: {os.path.join(os.getcwd(), 'results', 'dataset_processed.csv')}")

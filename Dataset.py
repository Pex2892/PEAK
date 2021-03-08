import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, f: str, sep: str = '', skiprows: int = 0):
        self.dataset = None
        self.filename = f
        self.type_db = os.path.splitext(os.path.join(os.getcwd(), 'dataset', self.filename))[1]
        self.sep = sep
        self.skiprows = skiprows

    def load(self):
        if os.path.exists(os.path.join(os.getcwd(), 'dataset', self.filename)):
            if self.type_db == '.csv':
                print('CSV')
            elif self.type_db == '.xlsx':
                self.dataset = pd.read_excel(os.path.join(os.getcwd(), 'dataset', self.filename))
                # return self.df
        else:
            print(f'The file "{self.filename}" was not found.')
            exit()

    def drop_cols(self, cols: list = None):
        if cols is None:

            # Calculates the frequency of NaN and divides it with respect to the total number of samples
            s_nan = self.dataset.isna().sum() / self.dataset.shape[0]

            # Takes columns that have a NaN frequency greater than 50% (0.5)
            cols_to_del = [i for i, v in s_nan.iteritems() if v > 0.5]

            # Drop specified labels from columns.
            self.dataset = self.dataset.drop(cols_to_del, axis=1)

            # Drop the rows where at least one element is missing.
            self.dataset = self.dataset.dropna()
        else:
            if len(cols) > 0:
                self.dataset = self.dataset.drop(cols, axis=1)

            else:
                print('Error: No column to delete was found')
                exit()

    def normalize(self, cols: list = None):
        min_max_scaler = MinMaxScaler()

        if cols is None:
            df_dtypes = self.dataset.select_dtypes(include=['int64', 'float64'])
            cols = df_dtypes.columns.values

        if len(cols) > 0:
            self.dataset[cols] = min_max_scaler.fit_transform(self.dataset[cols])
        else:
            print('Error: No column to normalize was found')
            exit()

    def categorial_to_numeric(self, cols: list = None, newcols: list = None):
        if cols is None:
            df_object = self.dataset.select_dtypes(include=['object'])
            cols = df_object.columns.values
            newcols = [f'{c}_factorized' for c in cols]

        if len(cols) == len(newcols):
            for c, nc in zip(cols, newcols):
                self.dataset.insert(self.dataset.columns.get_loc(c)+1, nc, pd.factorize(self.dataset[c])[0])

            # for c, nc in zip(cols, newcols):
            #     self.dataset[nc], unique = pd.factorize(self.dataset[c])

        else:
            print('The lengths between "cols" and "newcols" are not the same')
            exit()

    def split(self, X, Y, test_size, seed, shuffle):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed,
                                                            shuffle=shuffle)
        return [X_train, X_test, Y_train, Y_test]


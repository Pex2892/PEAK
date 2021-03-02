import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, f: str):
        self.filename = f

    def load(self):
        if os.path.exists(os.path.join(os.getcwd(), 'dataset', self.filename)):
            return pd.read_excel(os.path.join(os.getcwd(), 'dataset', self.filename))

        print(f'The file "{self.filename}" was not found.')
        exit()

    def drop_cols(self, df, cols):
        return df.drop(cols, axis=1)

    def normalize(self, df, cols):
        min_max_scaler = MinMaxScaler()
        df[cols] = min_max_scaler.fit_transform(df[cols])
        return df

    def categorial_to_numeric(self, df, cols, new_cols_name):
        uniques = []
        for c, nc in zip(cols, new_cols_name):
            df[nc], unique = pd.factorize(df[c])
            uniques.append(unique)
        return df, uniques

    def split(self, X, Y, test_size, seed, shuffle):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed,
                                                            shuffle=shuffle)
        return [X_train, X_test, Y_train, Y_test]


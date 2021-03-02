from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm


class Regression:
    def linear_regr(self, tt_set: list):
        model = LinearRegression(fit_intercept=True, normalize=True, copy_X=False, n_jobs=-1, positive=True)
        model = model.fit(tt_set[0], tt_set[2])  # passiamo i set di addestramento

        Y_pred = model.predict(tt_set[1])  # eseguiamo la predizione sul test set

        r = {
            'R2_score': sm.r2_score(tt_set[3], Y_pred),
            'Mean_absolute_error': sm.mean_absolute_error(tt_set[3], Y_pred),
            'Mean_squared_error': sm.mean_squared_error(tt_set[3], Y_pred),
            'Median_absolute_error': sm.median_absolute_error(tt_set[3], Y_pred),
            'Explain_variance_score': sm.explained_variance_score(tt_set[3], Y_pred),
            'Significance_model': self.evaluate_r2score(sm.r2_score(tt_set[3], Y_pred))
        }
        return r

    def evaluate_r2score(self, v: float):
        if v < 0.3:
            return 'The model is worthless!'
        elif 0.3 < v < 0.5:
            return 'The model is poor!'
        elif 0.5 < v < 0.7:
            return 'The model is discreet!'
        elif 0.7 < v < 0.9:
            return 'The model is good!'
        elif 0.9 < v < 1.0:
            return 'The model is great!'
        elif v == 1.0:
            return 'There is most likely an error in the model!'
        elif v < 0.0:
            return 'There is most likely an error!'

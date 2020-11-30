import joblib
import numpy as np
from sklearn import preprocessing

from classifiers.abs_classifier import ABSClassifier
from vpn_top_mid_fine import fine_to_mid


class LoadFineForMidScaled(ABSClassifier):
    def __init__(self):
        model_path = "/home/eyal/Documents/master/thesis/networking/NetML-Competition2020/results/_vpn2016_fine/XGBoostScaled_20201130-004305/pima.joblib.dat"
        clf = joblib.load(model_path)
        self.clf = clf

        self.scaler = preprocessing.StandardScaler() # TODO: should load also scaler

        self.is_trained = False # Should be true if also scaler was loaded

    def train(self, X_train, X_val, y_train, y_val):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.is_trained = True

        # Output accuracy of classifier
        print("Training Score: \t{:.5f}".format(self.clf.score(X_train_scaled, fine_to_mid(y_train))))
        X_val_scaled = self.scaler.transform(X_val)
        print("Validation Score: \t{:.5f}".format(self.clf.score(X_val_scaled, fine_to_mid(y_val))))

    def predict(self, X):
        if self.is_trained is False:
            print("WARN: RFScaled was not trained but predict was called")

        X_scaled = self.scaler.transform(X)
        ypred = fine_to_mid(self.clf.predict(X_scaled))

        return ypred

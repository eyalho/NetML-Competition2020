from sklearn import preprocessing
from xgboost import XGBClassifier

from classifiers.abs_classifier import ABSClassifier


class XGBoostScaledOptuna(ABSClassifier):
    def __init__(self):
        self.clf = XGBClassifier(booster="dart",
                                 alpha=2.1585186469130006e-06,
                                 max_depth=9,
                                 eta=1.2008186006089662e-05,
                                 gamma=2.6586554392733573e-06,
                                 grow_policy="lossguide",
                                 sample_type="uniform",
                                 normalize_type="forest",
                                 rate_drop=0.6820563384069672,
                                 skip_drop=0.05004706702962791,
                                 reg_lambda=0.33733524039826585)
        self.scaler = preprocessing.StandardScaler()

        self.is_trained = False

    def train(self, X_train, y_train, X_val=None, y_val=None):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.clf.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Output accuracy of classifier
        print("Training Score: \t{:.5f}".format(self.clf.score(X_train_scaled, y_train)))
        X_val_scaled = self.scaler.transform(X_val)
        print("Validation Score: \t{:.5f}".format(self.clf.score(X_val_scaled, y_val)))

    def predict(self, X):
        if self.is_trained is False:
            print("WARN: RFScaled was not trained but predict was called")

        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)

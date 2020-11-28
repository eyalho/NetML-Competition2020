from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn import preprocessing

from classifiers.abs_classifier import ABSClassifier


class TabNetScaled(ABSClassifier):
    def __init__(self):
        self.clf = TabNetClassifier()
        self.scaler = preprocessing.StandardScaler()
        self.is_trained = False

    def train(self, X_train, X_val, y_train, y_val):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.fit_transform(X_val)
        self.clf.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)])
        self.is_trained = True

    def predict(self, X):
        if self.is_trained is False:
            print("WARN: RFScaled was not trained but predict was called")

        X_scaled = self.scaler.fit_transform(X)
        return self.clf.predict(X_scaled)

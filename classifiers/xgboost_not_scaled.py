from sklearn import preprocessing
from xgboost import XGBClassifier

from classifiers.abs_classifier import ABSClassifier

class XGBoostNotScaled(ABSClassifier):
    def __init__(self):
        self.clf = XGBClassifier()
        self.is_trained = False

    def train(self, X_train, X_val, y_train, y_val):
        self.clf.fit(X_train, y_train)
        self.is_trained = True

        # Output accuracy of classifier
        print("Training Score: \t{:.5f}".format(self.clf.score(X_train, y_train)))
        print("Validation Score: \t{:.5f}".format(self.clf.score(X_val, y_val)))

    def predict(self, X):
        if self.is_trained is False:
            print("WARN: RFScaled was not trained but predict was called")
        return self.clf.predict(X)

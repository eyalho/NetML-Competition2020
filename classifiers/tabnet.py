from pytorch_tabnet.tab_model import TabNetClassifier

from classifiers.abs_classifier import ABSClassifier


class TabNet(ABSClassifier):
    def __init__(self):
        self.clf = TabNetClassifier()
        self.is_trained = False

    def train(self, X_train, X_val, y_train, y_val):
        self.clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        self.is_trained = True

    def predict(self, X):
        if self.is_trained is False:
            print("WARN: RFScaled was not trained but predict was called")
        return self.clf.predict(X)

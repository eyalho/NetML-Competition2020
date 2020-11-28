from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

from classifiers.abs_classifier import ABSClassifier


class RFScaled(ABSClassifier):
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
                                          max_features="auto")
        self.scaler = preprocessing.StandardScaler()

        self.is_trained = False

    def train(self, X_train, X_val, y_train, y_val):
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

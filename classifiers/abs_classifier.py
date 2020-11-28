from abc import abstractmethod, ABC


class ABSClassifier(ABC):
    @abstractmethod
    def train(self, X_train, X_val, y_train, y_val):
        pass

    @abstractmethod
    def predict(self, X):
        pass

from abc import abstractmethod, ABC


class ABSClassifier(ABC):
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

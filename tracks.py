import os
import time as t

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils.helper import get_submission_data, make_submission, plot_confusion_matrix, read_dataset


class Track:
    DATASETS_NAMES = ["cic2017", "netml", "vpn2016"]
    ANNO_LEVELS = ["top", "mid", "fine"]

    def __init__(self, dataset_name: str, anno_level: str):

        self.dataset_name = dataset_name
        self.anno_level = anno_level

        if anno_level not in self.ANNO_LEVELS:
            raise ValueError(f"{anno_level} not in {self.ANNO_LEVELS}")

        if dataset_name == self.DATASETS_NAMES[0]:
            self.dataset_path = "./data/CICIDS2017"
        elif dataset_name == self.DATASETS_NAMES[1]:
            self.dataset_path = "./data/NetML"
        elif dataset_name == self.DATASETS_NAMES[2]:
            self.dataset_path = "./data/non-vpn2016"
        else:
            raise ValueError(f"{dataset_name} not in {self.DATASETS_NAMES}")

        if anno_level == self.ANNO_LEVELS[1] and dataset_name == self.ANNO_LEVELS[0]:
            raise ValueError("cic2017 datasets cannot be trained with mid-level annotations. Use either top or fine.")
        if anno_level == self.ANNO_LEVELS[1] and dataset_name == self.ANNO_LEVELS[1]:
            raise ValueError("NetML datasets cannot be trained with mid-level annotations. Use either top or fine.")

        self.anno_level = anno_level

        self.training_set = os.path.join(self.dataset_path, "2_training_set")
        self.training_anno_file = os.path.join(self.dataset_path,
                                               f"2_training_annotations/2_training_anno_{self.anno_level}.json.gz")
        self.test_set = os.path.join(self.dataset_path, "1_test-std_set")
        self.challenge_set = os.path.join(self.dataset_path, "0_test-challenge_set")

        # Get training data in np.array format
        training_feature_names, ids, training_data, training_label, training_class_label_pair = read_dataset(
            self.training_set, self.training_anno_file, class_label_pairs=None)
        # Convert np.array to dataframe for easy manipulations
        training_df = pd.DataFrame(data=training_data,  # values
                                   index=[i for i in range(training_data.shape[0])],  # 1st column as index
                                   columns=training_feature_names)  # 1st row as the column names
        Xtrain = training_df.values

        self.Xtrain = Xtrain
        self.ytrain = training_label
        self.class_label_pair = training_class_label_pair
        self.Xtrain_ids = ids
        self.training_df = training_df

        self.class_names_list = list(sorted(self.class_label_pair.keys()))
        print(f"classes:{self.class_names_list}")

    def get_training_data(self):
        return self.Xtrain, self.ytrain, self.class_label_pair, self.Xtrain_ids

    def run_model_and_save_resutls(self, classifier=None):
        if classifier is None:
            classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
                                                max_features="auto")

        # Split validation set from training data
        X_train, X_val, y_train, y_val = train_test_split(self.Xtrain, self.ytrain,
                                                          test_size=0.2,
                                                          random_state=42,
                                                          stratify=self.ytrain)

        # Preprocess the data
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train classifier Model
        print("Training the model ...")
        classifier.fit(X_train_scaled, y_train)
        # classifier.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        # Output accuracy of classifier
        try:
            print("Training Score: \t{:.5f}".format(classifier.score(X_train_scaled, y_train)))
            print("Validation Score: \t{:.5f}".format(classifier.score(X_val_scaled, y_val)))
        except AttributeError:
            pass

        # Create folder for the results
        time_ = t.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(os.getcwd(), 'results', f"_{self.dataset_name}_{self.anno_level}", time_)
        os.makedirs(save_dir)

        # Plot normalized confusion matrix
        ypred = classifier.predict(X_val_scaled)
        np.set_printoptions(precision=2)
        plot_confusion_matrix(directory=save_dir, y_true=y_val, y_pred=ypred,
                              classes=self.class_names_list,
                              normalize=False)

    def plot_single_feature_distribution(self):
        pass

    def plot_several_features_umap_2d(self):
        pass


def submit(clf, test_set, scaler, class_label_pair, filepath):
    Xtest, ids = get_submission_data(test_set)
    X_test_scaled = scaler.transform(Xtest)
    print("Predicting on {} ...".format(test_set.split('/')[-1]))
    predictions = clf.predict(X_test_scaled)
    make_submission(predictions, ids, class_label_pair, filepath)


def main():
    for dataset in Track.DATASETS_NAMES:
        for level in Track.ANNO_LEVELS:
            if level == Track.ANNO_LEVELS[1] and dataset != Track.DATASETS_NAMES[2]:
                continue
            print(dataset, level)
            track = Track(dataset, level)
            track.run_model_and_save_resutls(classifier=None)


if __name__ == "__main__":
    main()

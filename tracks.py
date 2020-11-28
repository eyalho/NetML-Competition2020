import os
import time as t
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from classifiers.abs_classifier import ABSClassifier
from classifiers.tabnet_scaled import TabNetScaled
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

        self.counter_by_label_idx = dict(Counter(self.ytrain))
        self.counter_by_label_str = dict()
        for label_str, label_idx in self.class_label_pair.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            self.counter_by_label_str[label_str] = self.counter_by_label_idx[label_idx]

        self.class_names_list = list(sorted(self.class_label_pair.keys()))

        print("Loaded the dataset with:")
        for label_str, count in self.counter_by_label_str.items():
            print(f"{label_str}: {count}")

    def get_training_data(self):
        return self.Xtrain, self.ytrain, self.class_label_pair, self.Xtrain_ids

    def submit_test_std(self, classifier, save_dir):
        self._submit(classifier, self.test_set, self.class_label_pair, save_dir + "/submission_test-std.json")

    def submit_test_challenge(self, classifier, save_dir):
        self._submit(classifier, self.challenge_set, self.class_label_pair,
                     save_dir + "/submission_test-challenge.json")

    def _submit(self, classifier, test_set, class_label_pair, filepath):
        print("Predicting on {} ...".format(test_set.split('/')[-1]))
        Xtest, ids = get_submission_data(test_set)
        predictions = classifier.predict(Xtest)
        make_submission(predictions, ids, class_label_pair, filepath)

    def train_classifier_and_save_training_results(self, classifier: ABSClassifier):

        # Split validation set from training data
        X_train, X_val, y_train, y_val = train_test_split(self.Xtrain, self.ytrain,
                                                          test_size=0.2,
                                                          random_state=42,
                                                          stratify=self.ytrain)
        print("Training the model ...")
        classifier.train(X_train, X_val, y_train, y_val)

        # Create folder for the results
        time_ = t.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(os.getcwd(), 'results', f"_{self.dataset_name}_{self.anno_level}",
                                f"{classifier.__class__.__name__}_{time_}")
        os.makedirs(save_dir)

        # Plot normalized confusion matrix
        ypred = classifier.predict(X_val)
        np.set_printoptions(precision=2)
        plot_confusion_matrix(directory=save_dir, y_true=y_val, y_pred=ypred,
                              classes=self.class_names_list,
                              normalize=False)

        self.submit_test_challenge(classifier, save_dir)
        self.submit_test_challenge(classifier, save_dir)

    def plot_single_feature_distribution(self):
        pass

    def plot_several_features_umap_2d(self):
        pass


def main():
    # from pytorch_tabnet.tab_model import TabNetClassifier
    # clf = TabNetClassifier()
    track = Track("vpn2016", "top")
    classifier = TabNetScaled()
    track.train_classifier_and_save_training_results(classifier)

    for dataset in Track.DATASETS_NAMES:
        for level in Track.ANNO_LEVELS:
            if level == Track.ANNO_LEVELS[1] and dataset != Track.DATASETS_NAMES[2]:
                continue
            print(dataset, level)
            track = Track(dataset, level)
            track.train_classifier_and_save_training_results(classifier=classifier)


if __name__ == "__main__":
    main()

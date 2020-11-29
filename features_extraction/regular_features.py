import numpy as np
from pandas import DataFrame

from features_extraction.abs_features_extraction import ABSFeatureExtraction


class RegularFeatures(ABSFeatureExtraction):
    def extract(self, df: DataFrame) -> np.array:
        return df.values

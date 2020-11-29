import numpy as np
from pandas import DataFrame

from features_extraction.abs_features_extraction import ABSFeatureExtraction


class HistRemoverFeatures(ABSFeatureExtraction):

    def modify_df(self, df: DataFrame):
        OF = self.ORIGINAL_FEATURES

        delete_features = OF[OF.index("hdr_ccnt_0"):OF.index("hdr_ccnt_11") + 1]
        delete_features += OF[OF.index("intervals_ccnt_0"):OF.index("intervals_ccnt_15") + 1]
        delete_features += OF[OF.index("pld_ccnt_0"):OF.index("pld_ccnt_15") + 1]

        delete_features += OF[OF.index("rev_hdr_ccnt_0"):OF.index("rev_hdr_ccnt_11") + 1]
        delete_features += OF[OF.index("rev_intervals_ccnt_0"):OF.index("rev_intervals_ccnt_15") + 1]
        delete_features += OF[OF.index("rev_pld_ccnt_0"):OF.index("rev_pld_ccnt_15") + 1]

        for del_feature in delete_features:
            df.drop(del_feature, axis=1, inplace=True)

    def extract(self, df: DataFrame) -> np.array:
        self.modify_df(df)
        return df.values

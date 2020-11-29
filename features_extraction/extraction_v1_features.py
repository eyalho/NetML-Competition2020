import numpy as np
from pandas import DataFrame

from features_extraction.hist_remover_features import HistRemoverFeatures


class ExtractionV1Features(HistRemoverFeatures):
    def modify_df(self, df):
        super().modify_df(df)
        # ratio
        df['bytes_in_out_ratio'] = df['bytes_in'] / df['bytes_out']
        df['num_pkts_in'] = df['num_pkts_in'] / df['num_pkts_out']

        # rate
        df['bytes_in_rate'] = df['bytes_in'] / df['time_length']
        df['bytes_out_rate'] = df['bytes_out'] / df['time_length']
        df['num_pkts_in_rate'] = df['num_pkts_in'] / df['time_length']
        df['num_pkts_out_rate'] = df['num_pkts_out'] / df['time_length']

        # avg packet
        df['average_pkt_in_size'] = df['bytes_in'] / df['num_pkts_in']
        df['average_pkt_out_size'] = df['bytes_out'] / df['num_pkts_out']
        df['average_dt_in_size'] = df['time_length'] / df['num_pkts_in']
        df['average_dt_out_size'] = df['time_length'] / df['num_pkts_out']

        # ratio avg packets
        df['average_pkt_in_size_average_pkt_out_size_ratio'] = df['average_pkt_in_size'] / df['average_pkt_out_size']
        df['average_dt_in_size_average_dt_out_size_ratio'] = df['average_dt_in_size'] / df['average_dt_out_size']

    def extract(self, df: DataFrame) -> np.array:
        self.modify_df(df)
        X = df.values
        X = np.nan_to_num(X.astype(np.float32))
        return X

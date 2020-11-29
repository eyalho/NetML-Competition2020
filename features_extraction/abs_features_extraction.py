from abc import abstractmethod, ABC

import numpy as np
from pandas import DataFrame


class ABSFeatureExtraction(ABC):
    ORIGINAL_FEATURES = ['ack_psh_rst_syn_fin_cnt_0',
                         'ack_psh_rst_syn_fin_cnt_1',
                         'ack_psh_rst_syn_fin_cnt_2',
                         'ack_psh_rst_syn_fin_cnt_3',
                         'ack_psh_rst_syn_fin_cnt_4',
                         'bytes_in',
                         'bytes_out',
                         'dst_port',
                         'hdr_bin_40',
                         'hdr_ccnt_0',
                         'hdr_ccnt_1',
                         'hdr_ccnt_2',
                         'hdr_ccnt_3',
                         'hdr_ccnt_4',
                         'hdr_ccnt_5',
                         'hdr_ccnt_6',
                         'hdr_ccnt_7',
                         'hdr_ccnt_8',
                         'hdr_ccnt_9',
                         'hdr_ccnt_10',
                         'hdr_ccnt_11',
                         'hdr_distinct',
                         'hdr_mean',
                         'intervals_ccnt_0',
                         'intervals_ccnt_1',
                         'intervals_ccnt_2',
                         'intervals_ccnt_3',
                         'intervals_ccnt_4',
                         'intervals_ccnt_5',
                         'intervals_ccnt_6',
                         'intervals_ccnt_7',
                         'intervals_ccnt_8',
                         'intervals_ccnt_9',
                         'intervals_ccnt_10',
                         'intervals_ccnt_11',
                         'intervals_ccnt_12',
                         'intervals_ccnt_13',
                         'intervals_ccnt_14',
                         'intervals_ccnt_15',
                         'num_pkts_in',
                         'num_pkts_out',
                         'pld_bin_inf',
                         'pld_ccnt_0',
                         'pld_ccnt_1',
                         'pld_ccnt_2',
                         'pld_ccnt_3',
                         'pld_ccnt_4',
                         'pld_ccnt_5',
                         'pld_ccnt_6',
                         'pld_ccnt_7',
                         'pld_ccnt_8',
                         'pld_ccnt_9',
                         'pld_ccnt_10',
                         'pld_ccnt_11',
                         'pld_ccnt_12',
                         'pld_ccnt_13',
                         'pld_ccnt_14',
                         'pld_ccnt_15',
                         'pld_distinct',
                         'pld_max',
                         'pld_mean',
                         'pld_median',
                         'pr',
                         'rev_ack_psh_rst_syn_fin_cnt_0',
                         'rev_ack_psh_rst_syn_fin_cnt_1',
                         'rev_ack_psh_rst_syn_fin_cnt_2',
                         'rev_ack_psh_rst_syn_fin_cnt_3',
                         'rev_ack_psh_rst_syn_fin_cnt_4',
                         'rev_hdr_bin_40',
                         'rev_hdr_ccnt_0',
                         'rev_hdr_ccnt_1',
                         'rev_hdr_ccnt_2',
                         'rev_hdr_ccnt_3',
                         'rev_hdr_ccnt_4',
                         'rev_hdr_ccnt_5',
                         'rev_hdr_ccnt_6',
                         'rev_hdr_ccnt_7',
                         'rev_hdr_ccnt_8',
                         'rev_hdr_ccnt_9',
                         'rev_hdr_ccnt_10',
                         'rev_hdr_ccnt_11',
                         'rev_hdr_distinct',
                         'rev_intervals_ccnt_0',
                         'rev_intervals_ccnt_1',
                         'rev_intervals_ccnt_2',
                         'rev_intervals_ccnt_3',
                         'rev_intervals_ccnt_4',
                         'rev_intervals_ccnt_5',
                         'rev_intervals_ccnt_6',
                         'rev_intervals_ccnt_7',
                         'rev_intervals_ccnt_8',
                         'rev_intervals_ccnt_9',
                         'rev_intervals_ccnt_10',
                         'rev_intervals_ccnt_11',
                         'rev_intervals_ccnt_12',
                         'rev_intervals_ccnt_13',
                         'rev_intervals_ccnt_14',
                         'rev_intervals_ccnt_15',
                         'rev_pld_bin_128',
                         'rev_pld_ccnt_0',
                         'rev_pld_ccnt_1',
                         'rev_pld_ccnt_2',
                         'rev_pld_ccnt_3',
                         'rev_pld_ccnt_4',
                         'rev_pld_ccnt_5',
                         'rev_pld_ccnt_6',
                         'rev_pld_ccnt_7',
                         'rev_pld_ccnt_8',
                         'rev_pld_ccnt_9',
                         'rev_pld_ccnt_10',
                         'rev_pld_ccnt_11',
                         'rev_pld_ccnt_12',
                         'rev_pld_ccnt_13',
                         'rev_pld_ccnt_14',
                         'rev_pld_ccnt_15',
                         'rev_pld_distinct',
                         'rev_pld_max',
                         'rev_pld_mean',
                         'rev_pld_var',
                         'src_port',
                         'time_length']

    @abstractmethod
    def extract(self, df: DataFrame) -> np.array:
        pass

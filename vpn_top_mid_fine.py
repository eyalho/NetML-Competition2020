import numpy as np

top_counter  = dict(P2P=832, audio=90664, chat=6295, email=4159, file_transfer=25802, tor=122, video=3191)
mid_counter  = dict(aim=337, email=4159, facebook=35532, ftps=665, gmail=363, google=3, hangouts=37980, icq=352, netflix=274, scp=150, sftp=151, skype=46711, spotify=182, torrent=832, twitter=4, vimeo=365, voipbuster=2349, youtube=656)
fine_counter = dict(aim_chat=337, email=4159, facebook_audio=34782, facebook_chat=411, facebook_video=336, ftps_down=508, ftps_up=157, gmail_chat=363, hangouts_audio=36415, hangouts_chat=352, hangouts_video=1213, icq_chat=352, netflix=274, scp_down=76, scp_up=74, sftp_down=86, sftp_up=65, skype_audio=16936, skype_chat=4480, skype_file=24836, skype_video=459, spotify=182, tor_facebook=3, tor_google=3, tor_twitter=4, tor_vimeo=16, tor_youtube=96, torrent=832, vimeo=349, voipbuster=2349, youtube=560)

top_label_idx  = {'P2P': 0, 'audio': 1, 'chat': 2, 'email': 3, 'file_transfer': 4, 'tor': 5, 'video': 6}
mid_label_idx  = {'aim': 0, 'email': 1, 'facebook': 2, 'ftps': 3, 'gmail': 4, 'google': 5, 'hangouts': 6, 'icq': 7, 'netflix': 8, 'scp': 9, 'sftp': 10, 'skype': 11, 'spotify': 12, 'torrent': 13, 'twitter': 14, 'vimeo': 15, 'voipbuster': 16, 'youtube': 17}
fine_label_idx = {'aim_chat': 0, 'email': 1, 'facebook_audio': 2, 'facebook_chat': 3, 'facebook_video': 4, 'ftps_down': 5, 'ftps_up': 6, 'gmail_chat': 7, 'hangouts_audio': 8, 'hangouts_chat': 9, 'hangouts_video': 10, 'icq_chat': 11, 'netflix': 12, 'scp_down': 13, 'scp_up': 14, 'sftp_down': 15, 'sftp_up': 16, 'skype_audio': 17, 'skype_chat': 18, 'skype_file': 19, 'skype_video': 20, 'spotify': 21, 'tor_facebook': 22, 'tor_google': 23, 'tor_twitter': 24, 'tor_vimeo': 25, 'tor_youtube': 26, 'torrent': 27, 'vimeo': 28, 'voipbuster': 29, 'youtube': 30}

top_idx_label  = {0: 'P2P', 1: 'audio', 2: 'chat', 3: 'email', 4: 'file_transfer', 5: 'tor', 6: 'video'}
mid_idx_label  = {0: 'aim', 1: 'email', 2: 'facebook', 3: 'ftps', 4: 'gmail', 5: 'google', 6: 'hangouts', 7: 'icq', 8: 'netflix', 9: 'scp', 10: 'sftp', 11: 'skype', 12: 'spotify', 13: 'torrent', 14: 'twitter', 15: 'vimeo', 16: 'voipbuster', 17: 'youtube'}
fine_idx_label = {0: 'aim_chat', 1: 'email', 2: 'facebook_audio', 3: 'facebook_chat', 4: 'facebook_video', 5: 'ftps_down', 6: 'ftps_up', 7: 'gmail_chat', 8: 'hangouts_audio', 9: 'hangouts_chat', 10: 'hangouts_video', 11: 'icq_chat', 12: 'netflix', 13: 'scp_down', 14: 'scp_up', 15: 'sftp_down', 16: 'sftp_up', 17: 'skype_audio', 18: 'skype_chat', 19: 'skype_file', 20: 'skype_video', 21: 'spotify', 22: 'tor_facebook', 23: 'tor_google', 24: 'tor_twitter', 25: 'tor_vimeo', 26: 'tor_youtube', 27: 'torrent', 28: 'vimeo', 29: 'voipbuster', 30: 'youtube'}

fine_to_mid_dict = dict(aim_chat="aim", email="email", facebook_audio="facebook", facebook_chat="facebook", facebook_video="facebook", ftps_down="ftps", ftps_up="ftps", gmail_chat="gmail", hangouts_audio="hangouts", hangouts_chat="hangouts", hangouts_video="hangouts", icq_chat="icq", netflix="netflix", scp_down="scp", scp_up="scp", sftp_down="sftp", sftp_up="sftp", skype_audio="skype", skype_chat="skype", skype_file="skype", skype_video="skype", spotify="spotify", tor_facebook="facebook", tor_google="google", tor_twitter="twitter", tor_vimeo="vimeo", tor_youtube="youtube", torrent="torrent", vimeo="vimeo", voipbuster="voipbuster", youtube="youtube")
fine_to_top_dict = dict(aim_chat="chat", email="email", facebook_audio="audio", facebook_chat="chat", facebook_video="video", ftps_down="file_transfer", ftps_up="file_transfer", gmail_chat="chat", hangouts_audio="audio", hangouts_chat="chat", hangouts_video="video", icq_chat="chat", netflix="video", scp_down="file_transfer", scp_up="file_transfer", sftp_down="file_transfer", sftp_up="file_transfer", skype_audio="audio", skype_chat="chat", skype_file="file_transfer", skype_video="video", spotify="audio", tor_facebook="tor", tor_google="tor", tor_twitter="tor", tor_vimeo="tor", tor_youtube="tor", torrent="P2P", vimeo="video", voipbuster="audio", youtube="video")


def fine_to_top(ypred):
    new_pred = [top_label_idx[fine_to_top_dict[fine_idx_label[p]]] for p in ypred]
    return np.array(new_pred)


def fine_to_mid(ypred):
    new_pred = [mid_label_idx[fine_to_mid_dict[fine_idx_label[p]]] for p in ypred]
    return np.array(new_pred)

def test_top_from_fine_dict():
    # print("-" * 30)
    # print("--top_counter--")
    tot_v_top = 0
    for k, v in top_counter.items():
        # print(k, v)
        tot_v_top += v
    # print(tot_v_top)

    # print("-" * 30)
    # print("--top_to_fine--")
    from collections import defaultdict

    top_from_fine = defaultdict(lambda: 0)
    for k_fine, val_fine in fine_counter.items():
        top_from_fine[fine_to_top_dict[k_fine]] += val_fine
    tot_v_top_to_fine = 0
    for k, v in top_from_fine.items():
        tot_v_top_to_fine += v
        # print(k, v)
    # print(tot_v_top_to_fine)
    # print("-" * 30)

    assert tot_v_top_to_fine == tot_v_top
    assert top_from_fine == top_counter

def test_mid_from_fine_dict():
    # print("-" * 30)
    # print("--mid_counter--")
    tot_v_mid = 0
    for k, v in mid_counter.items():
        # print(k, v)
        tot_v_mid += v
    # print(tot_v_mid)

    # print("-" * 30)
    # print("--mid_to_fine--")
    from collections import defaultdict
    mid_from_fine = defaultdict(lambda: 0)
    for k_fine, val_fine in fine_counter.items():
        mid_from_fine[fine_to_mid_dict[k_fine]] += val_fine
    tot_v_mid_to_fine = 0
    for k, v in mid_from_fine.items():
        tot_v_mid_to_fine += v
        # print(k, v)
    # print(tot_v_mid_to_fine)
    # print("-" * 30)

    assert tot_v_mid_to_fine == tot_v_mid
    assert mid_from_fine == mid_counter


# from utils.helper import read_dataset
# training_set_foldername = "./data/non-vpn2016/2_training_set"
# anno_file_name = "./data/non-vpn2016/2_training_annotations/2_training_anno_mid.json.gz"
# training_feature_names, ids, training_data, training_label, training_class_label_pair = read_dataset(
#     training_set_foldername, anno_file_name, class_label_pairs=None)
# print(training_class_label_pair)
# # 0 -> aim_chat -> chat -> 2
# print("fine_idx", "0")
# print("fine_counter", fine_idx_label[0])
# print("top_counter", fine_to_top_dict[fine_idx_label[0]])
# print("top_idx", top_label_idx[fine_to_top_dict[fine_idx_label[0]]])
#
# print(top_label_idx[fine_to_top_dict[fine_idx_label[0]]])
#
# preds= [1,2,1,1,2,2,3,4,6,12]
# new_preds = [top_label_idx[fine_to_top_dict[fine_idx_label[p]]] for p in preds]
# print(new_preds)
# def reverse_dict(old_dict):
#     return dict([(value, key) for key, value in old_dict.items()])

if __name__ == "__main__":
    test_top_from_fine_dict()
    test_mid_from_fine_dict()


import os
import numpy as np
import pandas as pd
import mne
from autoreject import AutoReject
import scipy.io as scio
from src.configs.config import CFGLog
from src.preprocessing.utils import *

RESAMPLE_FREQ = CFGLog["parameters"]["sfreq"]
RANDOM_STATE = CFGLog["parameters"]["random_seed"]

def delete_multiple_elements(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


def get_stim_epochs(epochs, val_list, key_list, events_from_annot_drop_repeats_list,
                    min_dur_stim=320, max_dur_stim=1800, gap_ITI=100):
    stim_labels = []
    StimOn_ids = []
    key_wo_pp_ids = []
    key_wo_pp_lbls = []
    key_wo_pp_samps_to_ms = []
    key_to_pp_lag = []
    pp_updown_dur = []
    ITI_stim_gap = []

    for i in range(len(epochs) - 1):
        curr_pos = val_list.index(events_from_annot_drop_repeats_list[i][-1])
        curr_key_str = key_list[curr_pos]
        curr_val = val_list[curr_pos]

        next_pos = val_list.index(events_from_annot_drop_repeats_list[i + 1][-1])
        next_key_str = key_list[next_pos]
        next_val = val_list[next_pos]

        if (10 in val_list or 12 in val_list) and 3 in val_list:
            if (curr_val in range(3, 9)) and (next_val in range(10, 14)):
                StimOn_ids.append(i + 1)
                stim_labels.append(curr_key_str)
                key_to_pp_lag.append(
                    (events_from_annot_drop_repeats_list[i + 1][0] - events_from_annot_drop_repeats_list[i][0]) * SAMPS_TO_MS
                )

            elif (curr_val in range(3, 9)) and (next_val not in range(10, 14)):
                key_wo_pp_ids.append(i)
                key_wo_pp_lbls.append(curr_key_str)
                key_wo_pp_samps_to_ms.append(events_from_annot_drop_repeats_list[i][0] * SAMPS_TO_MS)

        elif 10 not in val_list or 12 not in val_list:
            if curr_val in range(3, 9):
                StimOn_ids.append(i)
                stim_labels.append(curr_key_str)

        elif 3 not in val_list:
            if ((curr_val == 10 or curr_val == 12) and (next_val == 11 or next_val == 13)
                and ((events_from_annot_drop_repeats_list[i + 1][0] - events_from_annot_drop_repeats_list[i][0]) > min_dur_stim * MS_TO_SAMP)
                and ((events_from_annot_drop_repeats_list[i + 1][0] - events_from_annot_drop_repeats_list[i][0]) < max_dur_stim * MS_TO_SAMP)):
                StimOn_ids.append(i)
                pp_updown_dur.append(
                    (events_from_annot_drop_repeats_list[i + 1][0] - events_from_annot_drop_repeats_list[i][0]) * SAMPS_TO_MS
                )
                ITI_stim_gap.append(
                    (events_from_annot_drop_repeats_list[i][0] - events_from_annot_drop_repeats_list[i - 1][0]) * SAMPS_TO_MS
                )

    return stim_labels, StimOn_ids, key_wo_pp_ids, key_wo_pp_lbls, key_wo_pp_samps_to_ms, key_to_pp_lag, pp_updown_dur, ITI_stim_gap


def labels_to_keys(txt_labels_list, val_list):
    stim_labels = [0] * len(txt_labels_list)
    if 10 in val_list or 11 in val_list or 12 in val_list or 13 in val_list:
        for i in range(len(stim_labels)):
            if txt_labels_list[i] in yes_hand_pain_list:
                stim_labels[i] = 3
            elif txt_labels_list[i] in med_hand_pain_list:
                stim_labels[i] = 4
            elif txt_labels_list[i] in no_hand_pain_list:
                stim_labels[i] = 5
            elif txt_labels_list[i] in yes_back_pain_list:
                stim_labels[i] = 6
            elif txt_labels_list[i] in med_back_pain_list:
                stim_labels[i] = 7
            elif txt_labels_list[i] in no_back_pain_list:
                stim_labels[i] = 8
    else:
        for i in range(len(stim_labels)):
            if txt_labels_list[i] in yes_hand_pain_list:
                stim_labels[i] = 3
            elif txt_labels_list[i] in no_hand_pain_list:
                stim_labels[i] = 4
            elif txt_labels_list[i] in yes_back_pain_list:
                stim_labels[i] = 5
            elif txt_labels_list[i] in no_back_pain_list:
                stim_labels[i] = 6

    return [num for num in stim_labels if isinstance(num, int)]


def process_annotations(raw, custom_mapping):
    events_from_annot, event_dict = mne.events_from_annotations(raw, event_id=custom_mapping)
    key_list = list(event_dict.keys())
    val_list = list(event_dict.values())

    events_from_annot_new = events_from_annot.copy()
    merged_flag = 0

    for i in range(len(events_from_annot) - 1):
        if events_from_annot[i][0] == events_from_annot[i + 1][0] and events_from_annot[i][2] < 10:
            merged_flag = 1
            events_from_annot_new = np.delete(events_from_annot_new, i + 1, axis=0)

        elif events_from_annot[i][0] == events_from_annot[i + 1][0] and events_from_annot[i + 1][2] < 10:
            merged_flag = 1
            events_from_annot_new = np.delete(events_from_annot_new, i, axis=0)

    if merged_flag:
        events_from_annot = events_from_annot_new

    repeated_flag = 0
    repeated_count = 0
    events_from_annot_new = events_from_annot.copy()

    for i in range(len(events_from_annot_new) - 1):
        if events_from_annot[i][2] in range(3, 8) and events_from_annot[i + 1][2] in range(3, 8):
            if ((events_from_annot[i + 1][0] - events_from_annot[i][0]) < 1000 * MS_TO_SAMP
                and events_from_annot[i][2] == events_from_annot[i + 1][2]
                and events_from_annot[i + 2][2] == events_from_annot[i + 1][2]):
                repeated_flag = 1
                repeated_count += 1

    if repeated_flag:
        events_from_annot = events_from_annot_new

    return events_from_annot, event_dict, key_list, val_list


def create_epochs(raw, events_from_annot, event_dict, tmin, tmax):
    if 10 in event_dict.values() or 12 in event_dict.values():
        epochs = mne.Epochs(
            raw,
            events_from_annot,
            event_dict,
            tmin=tmin,
            tmax=tmax,
            proj=True,
            preload=True,
            event_repeated="merge",
            baseline=(0, 0),
        )
    else:
        epochs = mne.Epochs(
            raw,
            events_from_annot,
            event_dict,
            tmin=tmin + 0.2,
            tmax=tmax + 0.2,
            proj=True,
            preload=True,
            event_repeated="merge",
            baseline=(0, 0),
        )
    return epochs


def adjust_annotations_for_repeats(epochs, events_from_annot):
    epo_drop_arr = np.array(epochs.drop_log, dtype=object)
    repeated_ids = np.argwhere(epo_drop_arr)
    events_from_annot_drop_repeats_arr = np.delete(events_from_annot, repeated_ids, 0)
    events_from_annot_drop_repeats_list = events_from_annot_drop_repeats_arr.tolist()
    return events_from_annot_drop_repeats_list


def correct_missing_pinpricks(events_from_annot_drop_repeats_list, key_wo_pp_ids, StimOn_ids):
    key_wo_check = input("0 or 1: Is at least one of the missing PP(s) actually missing (instead of just a wrong/extra keypress)?")
    if key_wo_check == "1":
        for i in range(len(key_wo_pp_ids)):
            needs_pp_adjustment = input("0 or 1: Does this key w/o pp event need correction?")
            if needs_pp_adjustment == "1":
                events_from_annot_drop_repeats_list[key_wo_pp_ids[i]][0] += 200 * MS_TO_SAMP
                StimOn_ids.append(key_wo_pp_ids[i])
        StimOn_ids.sort()
    return events_from_annot_drop_repeats_list, StimOn_ids


def load_pain_ratings(data_path, sub_id):
    for file in os.listdir(data_path):
        if file.startswith(sub_id):
            edf_dir = file

    xfname = ""
    for file in os.listdir(data_path / edf_dir):
        if file.endswith(".xlsx"):
            xfname = file

    df = pd.read_excel(data_path / edf_dir / xfname, sheet_name=0)
    return df


def extract_ground_truth(df):
    lower_back_flag = 0
    ground_truth_hand = []
    ground_truth_back = []
    pain_ratings_hand = []
    pain_ratings_back = []

    try:
        if isinstance(df["Unnamed: 2"].tolist().index("LOWER BACK "), int):
            lower_back_flag = 1
            column_back_idx = df["Unnamed: 2"].tolist().index("LOWER BACK ")
            ground_truth_hand = df["PIN PRICK PAIN SCALE "][3:column_back_idx].tolist()
            pain_ratings_hand = df["Unnamed: 1"][3:column_back_idx].tolist()
            pain_ratings_back = df["Unnamed: 1"][column_back_idx:].tolist()

            ground_truth_hand_new = [3 if el == " 32 guage (3) " else 4 for el in ground_truth_hand]
            ground_truth_back = df["PIN PRICK PAIN SCALE "][column_back_idx:].tolist()
            ground_truth_back_new = [6 if el == " 32 guage (3) " else 8 for el in ground_truth_back]

    except ValueError:
        print("Lower back not found in excel sheet")

    if lower_back_flag:
        ground_truth = ground_truth_hand_new + ground_truth_back_new
        pain_ratings_lst = pain_ratings_hand + pain_ratings_back
    else:
        ground_truth = df["PIN PRICK PAIN SCALE "][3:].tolist()
        pain_ratings_lst = df["Unnamed: 1"][3:].tolist()

    return ground_truth, pain_ratings_lst


def clean_ground_truth(ground_truth, pain_ratings_lst):
    if np.any(np.isnan(ground_truth)):
        excel_nans = np.where(np.isnan(ground_truth))
        excel_nans = excel_nans[0].tolist()
        delete_multiple_elements(ground_truth, excel_nans)

        pain_nans = np.where(np.isnan(pain_ratings_lst))
        pain_nans = pain_nans[0].tolist()
        delete_multiple_elements(pain_ratings_lst, pain_nans)
    return ground_truth, pain_ratings_lst


def verify_lists_match(keys_from_annot, ground_truth, stim_epochs, StimOn_ids):
    mtch_ans = "Yes." if ground_truth == keys_from_annot else "No!"
    if mtch_ans == "No!":
        simple_issue_check = input("0 or 1: Does the issue appear to consist of single mismatches ONLY?\n")
        if simple_issue_check == "1":
            correct_single_mismatches(keys_from_annot, ground_truth, stim_epochs, StimOn_ids)
        else:
            print("\nPlease correct the issues manually using SigViewer, deleting epochs if necessary.")
    return mtch_ans


def correct_single_mismatches(keys_from_annot, ground_truth, stim_epochs, StimOn_ids):
    iss_ids = []
    for i, el in enumerate(keys_from_annot):
        try:
            mismatch = next(
                (idx, x, y) for idx, (x, y) in enumerate(zip(keys_from_annot, ground_truth)) if x != y
            )
        except StopIteration:
            print("No (more) mismatches found. Exiting loop.")
            break
        else:
            iss_i = mismatch[0]
            if iss_i in iss_ids:
                continue
            else:
                iss_ids.append(iss_i)
                if keys_from_annot[iss_i + 1:iss_i + 1 + 2] == ground_truth[iss_i + 1:iss_i + 1 + 2]:
                    keys_from_annot[iss_i] = mismatch[2]
                elif keys_from_annot[iss_i + 1:iss_i + 1 + 2] == ground_truth[iss_i:iss_i + 2]:
                    del keys_from_annot[iss_i]
                    del StimOn_ids[iss_i]
                    stim_epochs.drop(iss_i)
                elif keys_from_annot[iss_i + 1:iss_i + 1 + 2] == ground_truth[iss_i + 1 + 1:iss_i + 1 + 1 + 2]:
                    del ground_truth[iss_i]
                    del pain_ratings_lst[iss_i]
                else:
                    print("Unknown error, check manually. **May be more than one marker missing/extra.\n")

    print("\nAFTER CORRECTION:\n")
    print("FROM ANNOTATIONS:")
    print(keys_from_annot)
    print(len(keys_from_annot))
    print("GROUND TRUTH:")
    print(ground_truth)
    print(len(ground_truth))
    print("PAIN RATINGS:")
    print(pain_ratings_lst)
    print(len(pain_ratings_lst))


def save_processed_data(stim_epochs, epo_times, ground_truth, pain_ratings_lst, save_path, sub_id):
    print("\nSaving processed epochs...")
    save_fname = sub_id[:3] + "_preprocessed-epo"
    stim_epochs.save(save_path / (save_fname + ".fif"), verbose=True, overwrite=True)

    print("\nSaving drop_log as mat file...")
    mdic = {"drop_log": dropped_epochs_list}
    scio.savemat(save_path / (sub_id[:3] + "_drop_log.mat"), mdic)

    print("\nSaving epoch_times as mat file...")
    mdic = {"epo_times": epo_times}
    scio.savemat(save_path / (sub_id[:3] + "_epo_times.mat"), mdic)

    print("\nSaving stim_labels as mat file...")
    mdic = {"stim_labels": ground_truth}
    scio.savemat(save_path / (sub_id[:3] + "_stim_labels.mat"), mdic)

    print("\nSaving pain_ratings as mat file...\n")
    mdic = {"pain_ratings": pain_ratings_lst}
    scio.savemat(save_path / (sub_id[:3] + "_pain_ratings.mat"), mdic)
    
def to_epo(raw, sub_id, data_path, save_path):
    """
    Preprocess the cleaned -raw.fif to epoched -epo.fif.
    Removes noisy trials and trials with movement artifact.
    raw: gets raw from to_raw()
    save_path: save processed epoch info as .mat files
    """

    global yes_hand_pain_list, med_hand_pain_list, no_hand_pain_list, yes_back_pain_list, med_back_pain_list, no_back_pain_list
    global SAMPS_TO_MS, MS_TO_SAMP
    MS_TO_SAMP = 400 / 1000
    SAMPS_TO_MS = 1000 / 400

    custom_mapping = {
        "eyes closed": 1,
        "Trigger#1": 1,
        "EYES CLOSED": 1,
        "eyes open": 2,
        "eyes opened": 2,
        "Trigger#2": 2,
        "EYES OPEN": 2,
        "eyes openned": 2,
        "pinprick hand": 3,
        "hand pinprick": 3,
        "Yes Pain Hand": 3,
        "Trigger#3": 3,
        "HAND PINPRICK": 3,
        "hand 32 gauge pinprick": 3,
        "Yes Hand Pain": 3,
        "Hand YES Pain prick": 3,
        "Med Pain Hand": 4,
        "Med Hand Pain": 4,
        "Hand Medium Pain prick": 4,
        "No Pain Hand": 5,
        "hand plastic": 5,
        "plastic hand": 5,
        "Trigger#4": 5,
        "HAND PLASTIC": 5,
        "hand plastic filament": 5,
        "No Hand Pain": 5,
        "Hand NO Pain": 5,
        "pinprick back": 6,
        "back pinprick": 6,
        "Yes Pain Back": 6,
        "BACK  PINPRICK": 6,
        "BACK PINPRICK": 6,
        "Trigger#5": 6,
        "back 32 gauge pinprick": 6,
        "Yes Back Pain": 6,
        "Back YES Pain prick": 6,
        "Med Pain Back": 7,
        "Med Back Pain": 7,
        "Back Medium Pain prick": 7,
        "plastic back": 8,
        "back plastic": 8,
        "No Pain Back": 8,
        "BACK PLASTIC": 8,
        "Trigger#6": 8,
        "back plastic filament": 8,
        "No Back Pain": 8,
        "Back No Pain": 8,
        "stop": 9,
        "Stop": 9,
        "STOP": 9,
        "1000001": 10,
        "100160": 10,
        "100480": 10,
        "1000000": 10,
        "1000010": 11,
        "100048": 11,
        "1100001": 12,
        "100320": 12,
        "1100010": 13,
    }

    yes_hand_pain_list = list(custom_mapping.keys())[8:16]
    med_hand_pain_list = list(custom_mapping.keys())[16:19]
    no_hand_pain_list = list(custom_mapping.keys())[19:27]
    yes_back_pain_list = list(custom_mapping.keys())[27:36]
    med_back_pain_list = list(custom_mapping.keys())[36:39]
    no_back_pain_list = list(custom_mapping.keys())[39:47]

    events_from_annot, event_dict, key_list, val_list = process_annotations(raw, custom_mapping)
    times_tup, time_win_path = get_time_window(5)
    tmin, bmax, tmax = times_tup

    epochs = create_epochs(raw, events_from_annot, event_dict, tmin, tmax)
    events_from_annot_drop_repeats_list = adjust_annotations_for_repeats(epochs, events_from_annot)

    stim_labels, StimOn_ids, key_wo_pp_ids, key_wo_pp_lbls, key_wo_pp_samps_to_ms, key_to_pp_lag, pp_updown_dur, ITI_stim_gap = get_stim_epochs(
        epochs, val_list, key_list, events_from_annot_drop_repeats_list)

    stim_epochs = epochs[StimOn_ids]

    if key_wo_pp_ids:
        events_from_annot_drop_repeats_list, StimOn_ids = correct_missing_pinpricks(events_from_annot_drop_repeats_list, key_wo_pp_ids, StimOn_ids)

    df = load_pain_ratings(data_path, sub_id)
    ground_truth, pain_ratings_lst = extract_ground_truth(df)
    ground_truth, pain_ratings_lst = clean_ground_truth(ground_truth, pain_ratings_lst)

    keys_from_annot = labels_to_keys(stim_labels, val_list)
    verify_lists_match(keys_from_annot, ground_truth, stim_epochs, StimOn_ids)

    ar = AutoReject(random_state=RANDOM_STATE)
    _, reject_log = ar.fit_transform(stim_epochs, return_log=True)
    bad_epochs_bool = reject_log.bad_epochs.tolist()
    dropped_epochs_list = [i for i, val in enumerate(bad_epochs_bool) if val]

    stim_epochs.drop(dropped_epochs_list)
    epo_times = np.delete(epo_times, dropped_epochs_list, axis=0)
    delete_multiple_elements(ground_truth, dropped_epochs_list)
    delete_multiple_elements(keys_from_annot, dropped_epochs_list)
    delete_multiple_elements(pain_ratings_lst, dropped_epochs_list)

    save_processed_data(stim_epochs, epo_times, ground_truth, pain_ratings_lst, save_path, sub_id)

    return stim_epochs, epo_times, ground_truth, pain_ratings_lst

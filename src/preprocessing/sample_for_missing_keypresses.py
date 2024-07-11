# %% [markdown]
# # Preprocess continuous FIF data to epoched FIF data
# ###  Output: *-epo.fif, *-epo_times.mat, and *-stim_labels.mat

# %% [markdown]
# ##### Define functions for finding relevant events + event name dictionary

# %%
import os
import numpy as np
import mne
import scipy.io as scio
import pandas as pd
from autoreject import AutoReject


# %%
## define functions for extracting relevant epochs
def delete_multiple_elements(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)  ## define functions for extracting relevant epochs


def labels_to_keys(txt_labels_list, val_list):
    stim_labels = [0] * len(txt_labels_list)
    if 10 in val_list or 11 in val_list or 12 in val_list or 13 in val_list:
        for i in range(0, len(stim_labels)):
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
        for i in range(0, len(stim_labels)):
            if txt_labels_list[i] in yes_hand_pain_list:
                stim_labels[i] = 3
            elif txt_labels_list[i] in no_hand_pain_list:
                stim_labels[i] = 4
            elif txt_labels_list[i] in yes_back_pain_list:
                stim_labels[i] = 5
            elif txt_labels_list[i] in no_back_pain_list:
                stim_labels[i] = 6

    # extract only integer keys
    key_els = [num for num in stim_labels if isinstance(num, (int))]

    return key_els


## define label dictionary for epoch annotations
## different keys account for different naming conventions
## NOTE: the Trigger#X naming convention does not specify between hand and back stimulus
custom_mapping = {
    "eyes closed": 1,
    "Trigger#1": 1,
    "EYES CLOSED": 1,  # eyes closed
    "eyes open": 2,
    "eyes opened": 2,
    "Trigger#2": 2,
    "EYES OPEN": 2,
    "eyes openned": 2,  # eyes open
    "pinprick hand": 3,
    "hand pinprick": 3,
    "Yes Pain Hand": 3,
    "Trigger#3": 3,
    "HAND PINPRICK": 3,
    "hand 32 gauge pinprick": 3,
    "Yes Hand Pain": 3,
    "Hand YES Pain prick": 3,
    # highest intensity pain stimulus
    "Med Pain Hand": 4,
    "Med Hand Pain": 4,
    "Hand Medium Pain prick": 4,  # intermediate intensity pain stimulus (HAND)
    "No Pain Hand": 5,
    "hand plastic": 5,
    "plastic hand": 5,
    "Trigger#4": 5,
    "HAND PLASTIC": 5,
    "hand plastic filament": 5,
    "No Hand Pain": 5,
    "Hand NO Pain": 5,
    # sensory stimulus, no pain
    "pinprick back": 6,
    "back pinprick": 6,
    "Yes Pain Back": 6,
    "BACK  PINPRICK": 6,
    "BACK PINPRICK": 6,
    "Trigger#5": 6,
    "back 32 gauge pinprick": 6,
    "Yes Back Pain": 6,
    "Back YES Pain prick": 6,
    # highest intensity pain stimulus (BACK)
    "Med Pain Back": 7,
    "Med Back Pain": 7,
    "Back Medium Pain prick": 7,  # intermediate intensity pain stimulus (BACK)
    "plastic back": 8,
    "back plastic": 8,
    "No Pain Back": 8,
    "BACK PLASTIC": 8,
    "Trigger#6": 8,
    "back plastic filament": 8,
    "No Back Pain": 8,
    "Back No Pain": 8,
    # sensory stimulus, no pain (BACK)
    "stop": 9,
    "Stop": 9,
    "STOP": 9,  # stop
    "1000001": 10,
    "100160": 10,
    "100480": 10,
    "1000000": 10,  # lesser weight pen tip down
    "1000010": 11,
    "100048": 11,  # lesser weight pen tip up
    "1100001": 12,
    "100320": 12,
    "1000000": 12,  # greater weight pen tip down
    "1100010": 13,  # greater weight pen tip up
}

# conversion factor for converting from given MS to SAMPLES
MS_TO_SAMP = 400 / 1000  # e.g. 300 ms * (400 Hz / 1000 ms) = 120 samples
SAMPS_TO_MS = 1000 / 400


# %%
def get_stim_epochs(
    epochs,
    val_list,
    key_list,
    events_from_annot_drop_repeats_list,
    min_dur_stim,
    max_dur_stim,
    gap_ITI,
):
    for i in range(len(epochs) - 1):
        # current epoch
        pre_curr_pos = val_list.index(
            events_from_annot_drop_repeats_list[i - 1][-1]
        )  # get position of epoch description value
        pre_curr_key_str = key_list[
            pre_curr_pos
        ]  # get key at position (e.g., 'Yes Pain Hand')
        pre_curr_val = val_list[pre_curr_pos]

        curr_pos = val_list.index(
            events_from_annot_drop_repeats_list[i][-1]
        )  # get position of epoch description value
        curr_key_str = key_list[curr_pos]  # get key at position (e.g., 'Yes Pain Hand')
        curr_val = val_list[curr_pos]

        next_pos = val_list.index(
            events_from_annot_drop_repeats_list[i + 1][-1]
        )  # get position of epoch description value
        next_key_str = key_list[next_pos]  # get key at position (e.g., 'Yes Pain Hand')
        next_val = val_list[next_pos]

        # for paradigms with NS, LS, HS pinprick keys_from_annot AND key presses
        if (10 in val_list or 12 in val_list) and 3 in val_list:
            # print(0)
            if (curr_val in range(3, 9)) and (next_val in range(10, 14)):
                # print('00')
                StimOn_ids.append(i + 1)  # save pinprick marker
                stim_labels.append(curr_key_str)  # save label
                key_to_pp_lag.append(
                    (
                        events_from_annot_drop_repeats_list[i + 1][0]
                        - events_from_annot_drop_repeats_list[i][0]
                    )
                    * SAMPS_TO_MS
                )
                # print(next_key_str)

            # check whether there are any pinprick keys_from_annot missing for some of the key presses
            elif (curr_val in range(3, 9)) and (next_val not in range(10, 14)):
                # print(curr_val, next_val)
                # print('01')
                key_wo_pp_ids.append(i)  # save pinprick marker
                key_wo_pp_lbls.append(curr_key_str)  # save label
                key_wo_pp_samps_to_ms.append(
                    events_from_annot_drop_repeats_list[i][0] * SAMPS_TO_MS
                )

                # print(next_key_str)

        # for paradigms with NS and HS, no LS. Key presses but no pinprick keys_from_annot
        elif 10 not in val_list or 12 not in val_list:
            # print(1)
            if curr_val in range(
                3, 9
            ):  # and curr_val <= 8 and curr_val != 4 and curr_val != 7:
                # print('11')
                StimOn_ids.append(i)
                stim_labels.append(curr_key_str)  # save label
                # key_to_pp_lag.append( (events_from_annot_drop_repeats_list[i+1][0] - events_from_annot_drop_repeats_list[i][0])*SAMPS_TO_MS )

        # for data missing all key presses, but has pinprick keys_from_annot
        elif 3 not in val_list:
            # print(2)
            # if current is pinprick down and next is pinprick up within:
            # max_dur_stim = 1500 # milliseconds
            # min_dur_stim =  344 # milliseconds
            # and if pinpricks occur at least gap_ITI apart:
            # gap_ITI = 6000 # milliseconds

            # if current is pinprick down and next is pinprick up within dur_stim:
            if (
                (curr_val == 10 or curr_val == 12)
                and (next_val == 11 or next_val == 13)
                and (
                    events_from_annot_drop_repeats_list[i + 1][0]
                    - events_from_annot_drop_repeats_list[i][0]
                )
                > float(min_dur_stim * MS_TO_SAMP)
                and (
                    events_from_annot_drop_repeats_list[i + 1][0]
                    - events_from_annot_drop_repeats_list[i][0]
                )
                < float(max_dur_stim * MS_TO_SAMP)
            ):  # and
                # AND if last pinprick marker is greater than gap_ITI before current marker:
                # (pre_curr_val in range(10,14)) and (curr_val in range(10,14)) and
                # ((events_from_annot_drop_repeats_list[i][0] - events_from_annot_drop_repeats_list[i-1][0]) > gap_ITI*MS_TO_SAMP ) ) :
                StimOn_ids.append(i)
                pp_updown_dur.append(
                    (
                        events_from_annot_drop_repeats_list[i + 1][0]
                        - events_from_annot_drop_repeats_list[i][0]
                    )
                    * SAMPS_TO_MS
                )
                ITI_stim_gap.append(
                    (
                        events_from_annot_drop_repeats_list[i][0]
                        - events_from_annot_drop_repeats_list[i - 1][0]
                    )
                    * SAMPS_TO_MS
                )

    return (
        stim_labels,
        StimOn_ids,
        key_wo_pp_ids,
        key_wo_pp_lbls,
        key_wo_pp_samps_to_ms,
        key_to_pp_lag,
        pp_updown_dur,
        ITI_stim_gap,
    )


# %%
sub_num = input("sub_num: ")

# %%
# import pain ratings to compare to annotations

eeg_dir = "../../Data/EEG DATA/"

for file in os.listdir(eeg_dir):
    if file.startswith(sub_num):
        edf_dir = file

xfname = ""
for file in os.listdir(eeg_dir + edf_dir):
    if file.endswith(".xlsx"):
        xfname = file

df = pd.read_excel((eeg_dir + edf_dir + "/" + xfname), sheet_name=0)

lower_back_flag = 0
try:
    if isinstance(df["Unnamed: 2"].tolist().index("LOWER BACK "), int):
        column_back_idx = df["Unnamed: 2"].tolist().index("LOWER BACK ")
        ground_truth_hand = df["PIN PRICK PAIN SCALE "][3:column_back_idx].tolist()

        pain_ratings_hand = df["Unnamed: 1"][3:column_back_idx].tolist()
        pain_ratings_back = df["Unnamed: 1"][column_back_idx:].tolist()

    if " 32 guage (3) " in ground_truth_hand:
        # index where rows switch to back
        ground_truth_hand_new = []
        for idx, el in enumerate(ground_truth_hand):
            if el == " 32 guage (3) ":
                ground_truth_hand_new.append(3)
            elif el == "PM (4)":
                ground_truth_hand_new.append(4)
        # back rows
        ground_truth_back = df["PIN PRICK PAIN SCALE "][column_back_id:].tolist()
        ground_truth_back_new = []
        for idx, el in enumerate(ground_truth_back):
            if el == " 32 guage (3) ":
                ground_truth_back_new.append(5)
            elif el == "PM (4)":
                ground_truth_back_new.append(6)

    lower_back_flag = 1
except:
    pass

if lower_back_flag:
    # concatenate lists
    ground_truth = ground_truth_hand_new + ground_truth_back_new
    pain_ratings_lst = pain_ratings_hand + pain_ratings_back
else:
    # concatenate lists
    ground_truth = df["PIN PRICK PAIN SCALE "][3:].tolist()
    pain_ratings_lst = df["Unnamed: 1"][3:].tolist()

# check if ground_truth contains nans which happens for some reason
if np.any(np.isnan(ground_truth)):
    excel_nans = np.where(np.isnan(ground_truth))
    excel_nans = excel_nans[0].tolist()
    delete_multiple_elements(ground_truth, excel_nans)

    pain_nans = np.where(np.isnan(pain_ratings_lst))
    pain_nans = pain_nans[0].tolist()
    delete_multiple_elements(pain_ratings_lst, pain_nans)

print(f"Loaded '{xfname}'!")

ground_truth = ground_truth

# %%
# Notes:
# C5 has a glitch where the 100048 keys_from_annot are all at time 0, so there are 6 key press keys_from_annot missing PP because they are followed by a 100048 instead of a 100480.
# C5 epoch correction sequence is (after 1 for starting correction): 1 0 1 1 1 1.

# %%
# subject ID
data_dir = "../../Data/Processed Data/"
save_dir = "../../Data/Processed Data/"

sub_id = ""
for file in os.listdir(data_dir):
    if file.startswith(sub_num) and file.endswith("_preprocessed-raw.fif"):
        sub_id = file

print(f"{sub_id}\nreading preprocessed-raw file...")
raw = mne.io.read_raw_fif(data_dir + sub_id, preload=True)

(events_from_annot, event_dict) = mne.events_from_annotations(
    raw, event_id=custom_mapping
)

# get key and val lists from event_dict
key_list = list(event_dict.keys())
val_list = list(event_dict.values())

raw

# %% [markdown]
# #### **ARE THERE ANY EPOCHS SHARING A SAMPLE INDEX WITH A KEYPRESS?**
# #### *IF so, delete the issue event prior/after key-press before instantiating Epochs object*

# %%
merged_flag = 0
events_from_annot_new = events_from_annot.copy()
for i in range(0, len(events_from_annot) - 1):
    # if any consecutive events occur at the same sample, delete the one thats not
    if (
        events_from_annot[i][0] == events_from_annot[i + 1][0]
        and events_from_annot[i][2] < 10
    ):
        merged_flag = 1
        print(
            f"Found merged epochs with labels {events_from_annot[i][2]} and {events_from_annot[i+1][2]}. Deleting epoch at index {i+1}."
        )
        print(f"{i}: {events_from_annot[i]}\n{i+1}: {events_from_annot [i+1]}")
        events_from_annot_new = np.delete(events_from_annot_new, i + 1, axis=0)

    elif (
        events_from_annot[i][0] == events_from_annot[i + 1][0]
        and events_from_annot[i + 1][2] < 10
    ):
        merged_flag = 1
        print(
            f"Found merged epochs with labels {events_from_annot[i][2]} and {events_from_annot[i+1][2]}. Deleting epoch at index {i}."
        )
        print(f"{i}: {events_from_annot[i]}\n{i+1}: {events_from_annot [i+1]}")
        events_from_annot_new = np.delete(events_from_annot_new, i, axis=0)

if merged_flag:
    events_from_annot = events_from_annot_new

# %% [markdown]
# #### **ARE THERE ANY REPEATED KEYPRESS keys_from_annot WITHIN THE SAME SECOND OR TWO?**
# #### *IF so, delete all prior issue events and just keep the last, then instantiate Epochs object*

# %%
repeated_flag = 0
repeated_count = 0
events_from_annot_new = events_from_annot.copy()
for i in range(0, len(events_from_annot_new) - 1):
    # if any consecutive events occur at the same sample, delete the one thats not
    if events_from_annot[i][2] in range(3, 8) and events_from_annot[i + 1][2] in range(
        3, 8
    ):
        # if the current and previous events have the same epoch and are less than a second apart BUT the following epoch is not less than a second apart:
        if (
            (
                (events_from_annot[i + 1][0] - events_from_annot[i][0])
                < 1000 * MS_TO_SAMP
            )
            and (events_from_annot[i][2] == events_from_annot[i + 1][2])
            and (events_from_annot[i + 2][2] == events_from_annot[i + 1][2])
        ):  # and not \
            # ( (events_from_annot[i+2][0] - events_from_annot[i+1][0]) > 1000*MS_TO_SAMP ) ):

            repeated_flag = 1
            print(
                f"Found repeated key press ({events_from_annot[i][2]}) at index {i}. Deleting epoch at index {i} and keeping the following epoch."
            )
            print(
                f"{np.round(events_from_annot[i][0]*SAMPS_TO_MS/1000,2)}: {events_from_annot[i]}"
            )
            print(
                f"{np.round(events_from_annot[i+1][0]*SAMPS_TO_MS/1000,2)}: {events_from_annot[i+1]}"
            )
            # events_from_annot_new = np.delete(events_from_annot_new, i+1, axis=0)
            repeated_count += 1

#         # else if
#         elif ( (events_from_annot[i+1][0] - events_from_annot[i][0]) < 1000*MS_TO_SAMP and \
#                (events_from_annot[i][2] == events_from_annot[i+1][2]) and not \
#                (events_from_annot[i+2][0] - events_from_annot[i+1][0]) > 1000*MS_TO_SAMP and \
#                (events_from_annot[i+2][2] != events_from_annot[i+1][2]) ):

if repeated_flag:
    events_from_annot = events_from_annot_new
    print(f"\nRemoved {repeated_count} extra keys_from_annot.")

# %% [markdown]
# Create initial epochs object with available keys_from_annot

# %%
import sys

sys.path.append("/home/wanglab/Documents/George Kenefati/Code/eeg_toolkit/")
# sys.path.append('/media/sb10flpc002/08e63286-43ce-4f61-9491-1ed048c96f20/Rachel Wu/eeg-projects/Code/eeg_toolkit/')
from eeg_toolkit import preprocess

# %%
times_tup, time_win_path = preprocess.get_time_window(5)
tmin, bmax, tmax = times_tup

# %%
# create events to epoch-ize data

# get key and val lists from event_dict
key_list = list(event_dict.keys())
val_list = list(event_dict.values())

# create epochs object differently depending on paradigm
if 10 in event_dict.values() or 12 in event_dict.values():
    print(f"{sub_id}\nCreating epochs WITH key presses\n")
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
    # when we don't have key presses, let's assume that the key press is 200 ms before the pinprick, as the tmin for the first case ^
    print(f"{sub_id}\nCreating epochs WITHOUT key presses\n")
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

# display.clear_output(wait=True)

epochs

# del raw # clear memory

# %%
# adjust events_from_annot for repeated events that are dropped by MNE
print(f"{sub_id}\nRemoving repeated epochs from annotations...")
epo_drop_arr = np.array(epochs.drop_log, dtype=object)
repeated_ids = np.argwhere(epo_drop_arr)
events_from_annot_drop_repeats_arr = np.delete(events_from_annot, repeated_ids, 0)
events_from_annot_drop_repeats_list = events_from_annot_drop_repeats_arr.tolist()
print(
    f"\nDropped {len(events_from_annot) - len(events_from_annot_drop_repeats_arr)} repeated epochs"
)
# display.clear_output(wait=True)

# %% [markdown]
# ##### Find stimulus events, labels, missing labels/samples, and PP/Stimulus Lags

# %%
# find epochs only from stim events
print(f"{sub_id}\nfinding the 60 pin-prick epochs...")

# get lists for keys and values of event_dict
key_list = list(event_dict.keys())
val_list = list(event_dict.values())

# intialize lists for epoch indices and labels
stim_labels = []
StimOn_ids = []
key_wo_pp_ids = []
key_wo_pp_lbls = []
key_wo_pp_samps_to_ms = []
key_to_pp_lag = []
pp_updown_dur = []
ITI_stim_gap = []  # uncertain whether this is calculated well enough to output it

# save only stimulus epochs
(
    stim_labels,
    StimOn_ids,
    key_wo_pp_ids,
    key_wo_pp_lbls,
    key_wo_pp_samps_to_ms,
    key_to_pp_lag,
    pp_updown_dur,
    _,
) = get_stim_epochs(
    epochs,
    val_list,
    key_list,
    events_from_annot_drop_repeats_list,
    # min_dur_stim = 100, max_dur_stim = 1900, gap_ITI = 100) # 044
    # min_dur_stim = 199, max_dur_stim = 1960, gap_ITI = 100) # 045
    min_dur_stim=255,
    max_dur_stim=1600,
    gap_ITI=100,
)  # 046
# min_dur_stim = 260, max_dur_stim = 1980, gap_ITI = 100) # C20
# min_dur_stim = 250, max_dur_stim = 1990, gap_ITI = 100) # C21
# min_dur_stim = 320, max_dur_stim = 1800, gap_ITI = 100) # C22
# min_dur_stim=320, max_dur_stim=1800, gap_ITI=100)  # C5.

stim_epochs = epochs[StimOn_ids]  # del epochs

if 3 not in val_list:
    print("LAGS BETWEEN PINPRICK UP AND DOWN:")
    print(pp_updown_dur)
    print(len(pp_updown_dur))

    # if ITI_stim_gap:
    #     print('LAGS BETWEEN STIMULUS EVENTS:')
    #     print(ITI_stim_gap)
    #     print(len(ITI_stim_gap))

stim_epochs

# %% [markdown]
# #### IF greater than 60, print below cell to compare to SigViewer.
# #### Then, if find an extra epoch, drop as usual

# %%
estimated_pp_times = [i[0] * SAMPS_TO_MS / 1000 for i in stim_epochs.events.tolist()]
print(len(estimated_pp_times), "\n")
[print(f"{el}") for i, el in enumerate(estimated_pp_times)]
# [print(f"{i}\t{el}") for i,el in enumerate(estimated_pp_times)];

# %%
# [print(f"{i}\t{el}") for i,el in enumerate(ground_truth)]

# %% [markdown]
# ##### Create label lists for each stimulus type

# %%
# define labels in separate lists
custom_mapping.keys()

# HAND
yes_hand_pain_list = list(custom_mapping.keys())[8:16]
# print(yes_hand_pain_list)
med_hand_pain_list = list(custom_mapping.keys())[16:19]
# print(med_hand_pain_list)
no_hand_pain_list = list(custom_mapping.keys())[19:27]
# print(no_hand_pain_list)


# BACK
yes_back_pain_list = list(custom_mapping.keys())[27:36]
# print(yes_back_pain_list)
med_back_pain_list = list(custom_mapping.keys())[36:39]
# print(med_back_pain_list)
no_back_pain_list = list(custom_mapping.keys())[39:47]
# print(no_back_pain_list)

# %% [markdown]
# ##### Create label array from annotations for comparison to ground truth (from Excel file)

# %%
# change labels to keys

keys_from_annot = labels_to_keys(stim_labels, val_list)

print("STIMULUS INDICES FROM ALL EVENTS:")
print(StimOn_ids)
print("\n\tLENGTH:", len(StimOn_ids))

print("\nSTIMULUS LABELS FROM ANNOTATIONS:")
print(stim_labels)
print("\n\tLENGTH:", len(stim_labels))

print("\nCONVERTED KEYS FROM LABELS:")
print(keys_from_annot)
print("\n\tLENGTH:", len(keys_from_annot))

if key_to_pp_lag:
    print("\nLAGS BETWEEN KEY PRESSES AND PINPRICKS:")
    print(key_to_pp_lag)
    from statistics import mean, stdev

    key_to_pp_lag_mean = mean(key_to_pp_lag)
    key_to_pp_lag_stdev = stdev(key_to_pp_lag)

    print(
        f"\n\tMean: {np.round(key_to_pp_lag_mean)} ms,  St Dev: {np.round(key_to_pp_lag_stdev)} ms"
    )

print("\n\tLENGTH:", len(key_to_pp_lag))

print("\nLABELS OF KEY PRESS WITHOUT PINPRICKS:")
print(key_wo_pp_lbls)

print("\nINDICES OF KEY PRESS WITHOUT PINPRICKS:")
print(key_wo_pp_ids)

print("\nTIME STAMPS OF KEY PRESS WITHOUT PINPRICKS (in seconds):")
print([np.round((i / 1000), 1) for i in key_wo_pp_samps_to_ms])
print("\n\tLENGTH:", len(key_wo_pp_samps_to_ms))

# %% [markdown]
# #### **Import stimulus and pain report information for the subject (from excel)**

# %%
print("FROM ANNOTATIONS:")
print(keys_from_annot)
print("LENGTH:", len(keys_from_annot))

print("\nGROUND TRUTH STIMULUS KEYS:")
print(ground_truth)
print("LENGTH:", len(ground_truth))

print("\nDo the lists match?")
mtch_ans = "Yes." if ground_truth == keys_from_annot else "No!"
print(mtch_ans)

# %% [markdown]
# #### Are there 5's that should be 4's, etc?

# %%
# keys_from_annot_new = keys_from_annot.copy()
# for i in range(0, len(keys_from_annot)):
#     if keys_from_annot[i] == 5:
#         keys_from_annot_new[i] = 4
#     elif keys_from_annot[i] == 8:
#         keys_from_annot_new[i] = 6
#     elif keys_from_annot[i] == 6:
#         keys_from_annot_new[i] = 5

# # validate first,
# print(keys_from_annot_new)
# print(len(keys_from_annot_new))

# # then uncomment and allow overwrite
# keys_from_annot = keys_from_annot_new

# %% [markdown]
# #### *IF missing back pinpricks (5 and 6), adjust here:*

# %%
# adjustment for Trigger# keys_from_annot that require changing 3 and 4 to 5 and 6 for back keys_from_annot
if 3 in keys_from_annot and (5 not in keys_from_annot or 6 not in keys_from_annot):
    back_switch_id = input(
        "Enter the index at which pinpricks switch to the lower back: "
    )
    keys_from_annot_new = keys_from_annot.copy()
    for i in range(int(back_switch_id), len(keys_from_annot)):
        if keys_from_annot[i] == 3:
            keys_from_annot_new[i] = 5
        elif keys_from_annot[i] == 4:
            keys_from_annot_new[i] = 6

    # validate first,
    print(keys_from_annot_new)
    print(len(keys_from_annot_new))

    # then uncomment and allow overwrite
    keys_from_annot = keys_from_annot_new

# %% [markdown]
# #### ***IF lists don't match, check for missing, extra, or point errors:***
# #### * *ONLY WORKS FOR SINGLE ERRORS* *
# #### * *If more errors exist after an 'Unknown Error' code, they will not be reported* *

# %%
# mismatch_list = []
# iss_ids = []

# if mtch_ans == 'No!':
#     simple_issue_check = input("0 or 1: Does the issue appear to consist of single mismatches ONLY?\n")
#     if simple_issue_check == '1':
#         for i, el in enumerate(keys_from_annot):
#             # find mismatch
#             try:
#                 mismatch = next( (idx, x, y) for idx, (x, y) in enumerate(zip(keys_from_annot, ground_truth)) if x!=y )
#             except:
#                 print('No (more) mismatches found. Exiting loop.')
#                 break
#             else:
#                 iss_i = mismatch[0]
#                 if iss_i in iss_ids: continue
#                 else:
#                     iss_ids.append( iss_i ); print(iss_i)
#                     if keys_from_annot[iss_i+1:iss_i+1+2] == ground_truth[iss_i+1:iss_i+1+2]:
#                         keys_from_annot[iss_i] = mismatch[2]
#                         print(f'Point error mismatch: changed label {mismatch[1]} to {mismatch[2]} in [keys_from_annot]\n')
#                     elif keys_from_annot[iss_i+1:iss_i+1+2] == ground_truth[iss_i:iss_i+2]:
#                         del keys_from_annot[iss_i]; del StimOn_ids[iss_i]; stim_epochs.drop(iss_i)
#                         print(f'Extra label mismatch: expected {mismatch[2]}, got {mismatch[1]}. Deleted label from [keys_from_annot].\n')
#                     elif keys_from_annot[iss_i+1:iss_i+1+2] == ground_truth[iss_i+1+1:iss_i+1+1+2]:
#                         del ground_truth[iss_i];
#                         print(f'Missing label mismatch: expected {mismatch[2]} in keys, deleted trial {iss_i} from [ground_truth] and from [txt_samps_list].\n')
#                     else:
#                         print(f'Unknown error, check manually. **May be more than one marker missing/extra.\n')
#                         # continue
#                         # break


#         print('\nAFTER CORRECTION:\n')

#         print('FROM ANNOTATIONS:')
#         print(keys_from_annot)
#         print(len(keys_from_annot))

#         print('GROUND TRUTH:')
#         print(ground_truth)
#         print(len(ground_truth))

#         print('\nDo the lists match now?')
#         mtch_ans = 'Yes.' if ground_truth == keys_from_annot else 'No!'
#         print(mtch_ans)
#     else:
#         print('\nPlease correct the issues manually using SigViewer, deleting epochs if necessary.')
# else:
#     print('Labels already match.')

# %% [markdown]
# ### Use this cell as a workspace if need to manually delete any epochs from stim_epochs and keys_from_annot:

# %%
### what to delete
drop_start = input("annotations drop START/LIST/NONE: ")
if "." not in drop_start and "," not in drop_start:
    drop_end = input("annotations drop END: ")
    drop_list = [*range(int(drop_start), int(drop_end) + 1)]
elif "," in drop_start:
    drop_list = [int(el) for el in drop_start.split(",")]
elif drop_start == ".":
    drop_list = []

# where to delete
stim_epochs.drop(drop_list)
delete_multiple_elements(StimOn_ids, drop_list)
delete_multiple_elements(keys_from_annot, drop_list)

print("len(epo_times):\t", len(StimOn_ids))

# %%
###########################################
# DROPPING FOR STIM LABELS AND PAIN RATINGS
###########################################
print("len(ground_truth):\t", len(ground_truth))
gt_drop_start = input("gt & pain ratings drop START: ")
if "." not in gt_drop_start and "," not in gt_drop_start:
    gt_drop_end = input("gt & pain ratings drop END: ")
    gt_drop = [*range(int(gt_drop_start), int(gt_drop_end) + 1)]
elif "," in gt_drop_start:
    gt_drop = [int(el) for el in gt_drop_start.split(",")]
elif gt_drop_start == ".":
    gt_drop = []

delete_multiple_elements(ground_truth, gt_drop)
delete_multiple_elements(pain_ratings_lst, gt_drop)
print("len(ground_truth):\t", len(ground_truth))

# %%
# # custom ground_truth for 044
# ground_truth = [4,5,4,5,5,4,5,5,4,4,8,8,7,7,8,7,7,8,7,8,5,5,
#                 5,5,5,6,6,8,8,6,8,6,8,6,8,3,3,4,4,3,4,4,3,4,
#                 3,7,6,6,7,6,6,7,6,7,7]
# print(len(ground_truth))

# pain_ratings_lst = [1,0,1,1,1,1,1,2,1,2,0,1,2,2,3,3,1,1,1,1,
#                     0,0,0,0,0,2,1,1,0,1,0,1,1,2,0,2,2,1,1,2,
#                     1,1,1,1,2,1,1,1,1,1,1,2,2,2,1]
# print(len(pain_ratings_lst))

# %% [markdown]
# ## Complete preprocessing and save

# %%
# verify stim_epochs object looks correct
print("FINAL EPOCH COUNT:", len(stim_epochs))
stim_epochs

# %% [markdown]
# #### Implement AutoReject, Save and Export

# %%
# Final check
epo_times = events_from_annot[StimOn_ids]

print("len(stim_epochs):\t", len(stim_epochs))

print("\nlen(ground_truth):\t", len(ground_truth))

print("\nlen(pain_ratings_lst):\t", len(pain_ratings_lst))

print("\nlen(epo_times):\t\t", len(epo_times))

# %%
# stim_epochs.drop_channels('Fp1')

# %%
# use autoreject package to automatically clean epochs
print(f"{sub_id}\nFinding epochs to clean...")
ar = AutoReject(random_state=42)
_, reject_log = ar.fit_transform(stim_epochs, return_log=True)
print(reject_log)
# display.clear_output(wait=True)

# drop rejected epochs
bad_epochs_bool = reject_log.bad_epochs.tolist()
dropped_epochs_list = [i for i, val in enumerate(bad_epochs_bool) if val]
print(f"Dropped {len(dropped_epochs_list)} epochs: ", dropped_epochs_list)

# save processed epochs
print("\nSaving processed epochs...")

save_fname = sub_id[:3] + "_preprocessed-epo"

stim_epochs.drop(dropped_epochs_list)
epo_times = np.delete(epo_times, dropped_epochs_list, axis=0)
delete_multiple_elements(ground_truth, dropped_epochs_list)
delete_multiple_elements(pain_ratings_lst, dropped_epochs_list)

# %%
# Final check
print("len(dropped_epochs_list):\t", len(dropped_epochs_list))
print("\nlen(stim_epochs):\t", len(stim_epochs))
print("\nlen(epo_times):\t\t", len(epo_times))
print("\nlen(ground_truth):\t", len(ground_truth))
print("\nlen(pain_ratings_lst):\t", len(pain_ratings_lst))

# %%
# Complete the saves
stim_epochs.save(data_dir + save_fname + ".fif", verbose=True, overwrite=True)

# save drop log
print("\nSaving drop_log as mat file...")
mdic = {"drop_log": dropped_epochs_list}
scio.savemat(data_dir + sub_id[:3] + "_drop_log.mat", mdic)

# save epo_times
print("\nSaving epoch_times as mat file...")
mdic = {"epo_times": epo_times}
scio.savemat(data_dir + sub_id[:3] + "_epo_times.mat", mdic)

# save stim labels
print("\nSaving stim_labels as mat file...")
mdic = {"stim_labels": ground_truth}
scio.savemat(data_dir + sub_id[:3] + "_stim_labels.mat", mdic)

# save pain ratings
print("\nSaving pain_ratings as mat file...\n")
mdic = {"pain_ratings": pain_ratings_lst}
scio.savemat(save_dir + sub_id[:3] + "_pain_ratings.mat", mdic)

print("Done.")
# display.clear_output(wait=True)

# %%


# %%

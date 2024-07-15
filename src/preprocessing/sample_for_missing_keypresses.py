

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

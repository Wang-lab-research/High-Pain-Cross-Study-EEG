import os
import numpy as np
import pandas as pd
import mne
from autoreject import AutoReject
from src.configs.config import CFGLog
from src.preprocessing import utils as pre_utils

RESAMPLE_FREQ = CFGLog["parameters"]["sfreq"]
RANDOM_STATE = CFGLog["parameters"]["random_seed"]


def get_stimulus_epochs(
    epochs,
    event_event_values,
    event_keys,
    cleaned_events,
    min_duration_stim=320,
    max_duration_stim=1800,
    gap_iti=100,
):
    """
    Find stimulus epochs in epochs data.

    Args:
        epochs (mne.Epochs): The epochs data.
        event_event_values (list): The event values.
        event_keys (list): The event keys.
        cleaned_events (list): The cleaned events.
        min_duration_stim (int): The minimum duration of stimulus in ms.
        max_duration_stim (int): The maximum duration of stimulus in ms.
        gap_iti (int): The intertrial interval.

    Returns:
        tuple: A tuple of stimulus events, key without pinprick events, key without pinprick samples in ms,
        key to pinprick lag in ms, pinprick up-down duration in ms, and ITI-stimulus gap in ms.
    """
    # Initialize variables to store results
    stimulus_events = []
    key_without_pp_events = []
    key_without_pp_samps_to_ms = []
    key_to_pp_lag = []
    pp_updown_duration = []
    iti_stim_gap = []

    # Define keyboard and pinprick ids
    keyboard_ids = range(3, 9)
    pinprick_ids = range(10, 14)

    # Iterate over epochs
    for i in range(len(epochs) - 1):
        # Find current and next event positions in event_event_values
        current_position = event_event_values.index(cleaned_events[i][-1])
        current_key_str = event_keys[current_position]
        current_value = event_event_values[current_position]

        next_position = event_event_values.index(cleaned_events[i + 1][-1])
        next_value = event_event_values[next_position]

        # Check if there is a stimulus event
        if (
            10 in event_event_values or 12 in event_event_values
        ) and 3 in event_event_values:
            # Check if current event is a keyboard event and next event is a pinprick event
            if current_value in keyboard_ids and next_value in pinprick_ids:
                # Add stimulus event to the list
                stimulus_events.append((i + 1, current_key_str))
                # Calculate key to pinprick lag in ms
                key_to_pp_lag.append(
                    (cleaned_events[i + 1][0] - cleaned_events[i][0]) * SAMPS_TO_MS
                )

            # Check if current event is a keyboard event and next event is not a pinprick event
            elif current_value in keyboard_ids and next_value not in pinprick_ids:
                # Add key without pinprick event to the list
                key_without_pp_events.append((i, current_key_str))
                # Add key without pinprick samples in ms to the list
                key_without_pp_samps_to_ms.append(cleaned_events[i][0] * SAMPS_TO_MS)

        # Check if there is no stimulus event
        elif 10 not in event_event_values or 12 not in event_event_values:
            # Check if current event is a keyboard event
            if current_value in keyboard_ids:
                # Add stimulus event to the list
                stimulus_events.append((i, current_key_str))

        # Check if there is no pinprick event
        elif 3 not in event_event_values:
            # Check if there is a stimulus event
            if (
                (current_value == 10 or current_value == 12)
                and (next_value == 11 or next_value == 13)
                and (
                    (cleaned_events[i + 1][0] - cleaned_events[i][0])
                    > min_duration_stim * MS_TO_SAMP
                )
                and (
                    (cleaned_events[i + 1][0] - cleaned_events[i][0])
                    < max_duration_stim * MS_TO_SAMP
                )
            ):
                # Add stimulus event to the list
                stimulus_events.append(i)
                # Calculate pinprick up-down duration in ms
                pp_updown_duration.append(
                    (cleaned_events[i + 1][0] - cleaned_events[i][0]) * SAMPS_TO_MS
                )
                # Calculate ITI-stimulus gap in ms
                iti_stim_gap.append(
                    (cleaned_events[i][0] - cleaned_events[i - 1][0]) * SAMPS_TO_MS
                )

    # Return the results
    return (
        stimulus_events,
        key_without_pp_events,
        key_without_pp_samps_to_ms,
        key_to_pp_lag,
        pp_updown_duration,
        iti_stim_gap,
    )


def get_stimulus_labels(stimulus_labels, event_values):
    """
    This function takes in a list of stimulus labels and a list of event values
    and returns a list of stimulus labels with their corresponding numerical values.

    Args:
        stimulus_labels (list): A list of stimulus labels.
        event_values (list): A list of event values.

    Returns:
        list: A list of stimulus labels with their corresponding numerical values.
    """

    # Initialize a list of zeros with the same length as the input list of stimulus labels.
    cleaned_stim_labels = [0] * stimulus_labels

    # Define a dictionary that maps stimulus labels to their corresponding numerical values.
    stimulus_types = {
        "high_hand": 3,  # High hand stimulus label mapped to numerical value 3.
        "med_hand": 4,  # Medium hand stimulus label mapped to numerical value 4.
        "low_hand": 5,  # Low hand stimulus label mapped to numerical value 5.
        "high_back": 6,  # High back stimulus label mapped to numerical value 6.
        "med_back": 7,  # Medium back stimulus label mapped to numerical value 7.
        "low_back": 8,  # Low back stimulus label mapped to numerical value 8.
    }

    # Check if any of the stimulus labels are present in the event values.
    if any(stimulus in event_values for stimulus in stimulus_types.keys()):
        # If any of the stimulus labels are present in the event values, update the stimulus_labels list accordingly.
        for i, label in enumerate(stimulus_labels):
            if label in stimulus_types:
                cleaned_stim_labels[i] = stimulus_types[label]
    else:
        # If none of the stimulus labels are present in the event values, update the stimulus_labels list accordingly.
        for i, label in enumerate(stimulus_labels):
            if label in ("high_hand", "low_hand"):
                cleaned_stim_labels[i] = (
                    3  # High hand and low hand stimulus labels mapped to numerical value 3.
                )
            elif label == "high_back":
                cleaned_stim_labels[i] = (
                    5  # High back stimulus label mapped to numerical value 5.
                )
            elif label == "low_back":
                cleaned_stim_labels[i] = (
                    6  # Low back stimulus label mapped to numerical value 6.
                )

    # Return a list of stimulus labels with their corresponding numerical values, excluding any labels that have a value of 0.
    return [label for label in cleaned_stim_labels if label != 0]


def process_annotations(raw, event_dict):
    """
    Process annotations from raw data to filter out duplicate and repeated events.

    Args:
        raw (mne.io.Raw): The raw data.
        event_dict (dict): The event dictionary.

    Returns:
        Tuple: The filtered events, event dictionary, event event_keys, and event event_values.
    """
    events, event_id = mne.find_events(raw, event_id=event_dict)
    event_keys = list(event_id.event_keys())
    event_values = list(event_id.event_values())

    events = remove_duplicate_events(events)
    events = remove_repeated_events(events)

    return events, event_dict, event_keys, event_values


def remove_duplicate_events(events):
    events_copy = events.copy()
    # Check if the next event is the same as the current and is a keyboard press (event ID less than 10)
    for i in range(len(events_copy) - 1):
        if events_copy[i][0] == events_copy[i + 1][0] and events_copy[i][2] < 10:
            events_copy = np.delete(events_copy, i + 1, axis=0)
    return events_copy


def remove_repeated_events(events):
    events_copy = events.copy()
    # Check if the next event is within 1 sec of the current and is the same event
    for i in range(len(events_copy) - 1):
        if events_copy[i][2] in range(3, 8) and events_copy[i + 1][2] in range(3, 8):
            if (
                (events_copy[i + 1][0] - events_copy[i][0]) < 1000 * MS_TO_SAMP
                and events_copy[i][2] == events_copy[i + 1][2]
                and events_copy[i + 2][2] == events_copy[i + 1][2]
            ):
                events_copy = np.delete(events_copy, i + 1, axis=0)
    return events_copy


def create_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_dict: dict,
    tmin: float,
    tmax: float,
) -> mne.Epochs:
    """
    Create epochs object from raw data and events.
    If the event dictionary contains event IDs 10 or 12, use the default time range.
    Otherwise, shift the time range by 0.2 seconds.

    Args:
        raw (mne.io.Raw): The raw data.
        events (np.ndarray): The event array.
        event_dict (dict): The event dictionary.
        tmin (float): The start time of the time range.
        tmax (float): The end time of the time range.

    Returns:
        mne.Epochs: The epochs object.
    """
    time_range = (
        (tmin, tmax)
        if 10 in event_dict.event_values() or 12 in event_dict.event_values()
        else (tmin + 0.2, tmax + 0.2)
    )
    epochs = mne.Epochs(
        raw,
        events,
        event_dict,
        tmin=time_range[0],
        tmax=time_range[1],
        proj=None,
        preload=True,
        event_repeated="merge",
        baseline=None,
    )
    return epochs


def remove_dropped_events(epochs, events):
    dropped_event_events = np.argwhere(np.array(epochs.drop_log, dtype=object))
    cleaned_events = np.delete(events, dropped_event_events, axis=0)
    return cleaned_events.tolist()


def correct_missing_pinpricks(events, indices_without_pp, stimulus_indices):
    should_correct_pinprick = (
        input(
            "Are any of the missing pinpricks actually missing (instead of just a wrong or extra keypress)?"
        ).lower()
        == "yes"
    )

    if should_correct_pinprick:
        for index in indices_without_pp:
            should_correct_key = (
                input(
                    f"Should the key without pinprick at index {index} be corrected?"
                ).lower()
                == "yes"
            )
            if should_correct_key:
                events[index][0] += 200 * MS_TO_SAMP
                stimulus_indices.append(index)
        stimulus_indices.sort()

    return events, stimulus_indices


def load_pain_ratings(data_dir, subject_id):
    edf_dir = next((f for f in os.listdir(data_dir) if subject_id in f), None)
    if edf_dir is None:
        raise ValueError(f"No directory found for subject {subject_id}")

    xlsx_file = next(
        (f for f in os.listdir(data_dir / edf_dir) if f.endswith(".xlsx")), None
    )
    if xlsx_file is None:
        raise ValueError(f"No XLSX file found for subject {subject_id}")

    data_path = data_dir / edf_dir / xlsx_file
    df = pd.read_excel(data_path, sheet_name=0)
    return df


def get_labels_and_ratings(data_frame):
    lower_back_found = False
    hand_labels = []
    back_labels = []
    hand_ratings = []
    back_ratings = []

    try:
        back_index = data_frame["Unnamed: 2"].tolist().index("LOWER BACK ")
        lower_back_found = True
        hand_labels = data_frame["PIN PRICK PAIN SCALE "][3:back_index].tolist()
        hand_ratings = data_frame["Unnamed: 1"][3:back_index].tolist()
        back_labels = data_frame["PIN PRICK PAIN SCALE "][back_index:].tolist()
        back_ratings = data_frame["Unnamed: 1"][back_index:].tolist()

        hand_labels = [3 if label == " 32 guage (3) " else 4 for label in hand_labels]
        back_labels = [6 if label == " 32 guage (3) " else 8 for label in back_labels]

    except ValueError:
        pass

    if lower_back_found:
        labels = hand_labels + back_labels
        ratings = hand_ratings + back_ratings
    else:
        labels = data_frame["PIN PRICK PAIN SCALE "][3:].tolist()
        ratings = data_frame["Unnamed: 1"][3:].tolist()

    # Remove NaNs
    gt_stim_labels = [el for el in labels if not np.isnan(el)]
    pain_ratings = [el for el in ratings if not np.isnan(el)]

    return gt_stim_labels, pain_ratings


def verify_lists_match(
    cleaned_stim_labels, ground_truth_labels, stim_epochs, stimulus_labels, pain_ratings
):
    mismatch_message = "No!" if ground_truth_labels != cleaned_stim_labels else "Yes."
    single_mismatch_issue = input(
        "Does the issue appear to consist of single mismatches ONLY? (0 or 1)\n"
    )
    if single_mismatch_issue == "1":
        (
            corrected_stim_labels,
            corrected_ground_truth_labels,
            corrected_stim_epochs,
            corrected_stimulus_labels,
            corrected_pain_ratings,
        ) = correct_single_mismatches(
            cleaned_stim_labels,
            ground_truth_labels,
            stim_epochs,
            stimulus_labels,
            pain_ratings,
        )
    else:
        raise ValueError(
            "Please correct the issues manually using SigViewer, deleting epochs if necessary."
        )
    return (
        corrected_stim_labels,
        corrected_ground_truth_labels,
        corrected_stim_epochs,
        corrected_stimulus_labels,
        corrected_pain_ratings,
    )


def correct_single_mismatches(
    stimulus_labels, ground_truth_labels, stim_epochs, pain_ratings
):
    corrected_ground_truth_labels = ground_truth_labels.copy()
    corrected_stimulus_labels = stimulus_labels.copy()
    corrected_pain_ratings = pain_ratings.copy()

    mismatches = []
    for i, stim_label in enumerate(corrected_stimulus_labels):
        try:
            mismatch = next(
                (idx, x, y)
                for idx, (x, y) in enumerate(
                    zip(corrected_stimulus_labels, corrected_ground_truth_labels)
                )
                if x != y
            )
        except StopIteration:
            print("No (more) mismatches found. Exiting loop.")
            break
        else:
            mismatches.append(mismatch)

    for mismatch in mismatches:
        index = mismatch[0]
        if index in [m[0] for m in mismatches]:
            continue

        if (
            corrected_stimulus_labels[index + 1 : index + 1 + 2]
            == corrected_ground_truth_labels[index + 1 : index + 1 + 2]
        ):
            corrected_stimulus_labels[index] = mismatch[2]
        elif (
            corrected_stimulus_labels[index + 1 : index + 1 + 2]
            == corrected_ground_truth_labels[index : index + 2]
        ):
            corrected_stimulus_labels.pop(index)
            corrected_stimulus_labels.pop(index)
            stim_epochs.drop(index)
        elif (
            corrected_stimulus_labels[index + 1 : index + 1 + 2]
            == corrected_ground_truth_labels[index + 1 + 1 : index + 1 + 1 + 2]
        ):
            corrected_ground_truth_labels.pop(index)
            corrected_pain_ratings.pop(index)
        else:
            raise ValueError("Unknown error, check manually.")

    print("\nAFTER CORRECTION:\n")
    print("FROM ANNOTATIONS:")
    print(corrected_stimulus_labels)
    print(len(corrected_stimulus_labels))
    print("GROUND TRUTH:")
    print(ground_truth_labels)
    print(len(ground_truth_labels))
    print("PAIN RATINGS:")
    print(pain_ratings)
    print(len(pain_ratings))

    return (
        corrected_stimulus_labels,
        corrected_ground_truth_labels,
        stim_epochs,
        corrected_pain_ratings,
    )


def zscore_epochs(epochs):
    epochs_data = epochs.get_data()
    zscores = (epochs_data - epochs_data.mean(axis=0)) / epochs_data.std(axis=0)
    info = epochs.info
    zscored_epochs = mne.EpochsArray(
        data=zscores, info=info, tmin=epochs.tmin, event_id=epochs.event_id
    )
    return zscored_epochs


def preprocess_epochs(raw, sub_id, data_path, TIME_RANGE, PERISTIM_TIME_WIN):
    """
    Preprocess epochs by correcting missing/incorrect/extra key-presses,
    classifying stimulus types, rejecting noisy trials, and verifying that the
    number of trials match remaining stimulus labels and pain ratings.

    Args:
        raw (mne.io.Raw): The raw data.
        sub_id (str): The subject ID.
        data_path (str): The path to the data.
        TIME_RANGE (tuple): The time range.
        PERISTIM_TIME_WIN (tuple): The persistence time window.
        BASELINE (tuple): The baseline time window.

    Returns:
        tuple: The preprocessed epochs, epoch times, ground truth labels, and
            pain ratings.
    """

    # Set global variables
    global high_hand, med_hand, low_hand, high_back, med_back, low_back
    global SAMPS_TO_MS, MS_TO_SAMP
    MS_TO_SAMP = RESAMPLE_FREQ / raw.info["sfreq"]
    SAMPS_TO_MS = raw.info["sfreq"] / RESAMPLE_FREQ

    # Load event dictionary
    event_dict = CFGLog["parameters"]["event_dict"]

    # Extract lists from event_dict which is a dictionary of dicts
    high_hand = list(event_dict["high_hand"].event_keys())
    med_hand = list(event_dict["med_hand"].event_keys())
    low_hand = list(event_dict["low_hand"].event_keys())
    high_back = list(event_dict["high_back"].event_keys())
    med_back = list(event_dict["med_back"].event_keys())
    low_back = list(event_dict["low_back"].event_keys())

    # Also combine event_dict dicts into one dict
    event_dict = {
        **event_dict["high_hand"],
        **event_dict["med_hand"],
        **event_dict["low_hand"],
        **event_dict["high_back"],
        **event_dict["med_back"],
        **event_dict["low_back"],
        **event_dict["stop"],
        **event_dict["pinprick_markers"],
    }

    # Get events and event_id, and remove duplicates
    events, event_id, event_keys, event_values = process_annotations(raw, event_dict)

    # Get desired time window
    times_tup, time_win_path = pre_utils.get_time_window(PERISTIM_TIME_WIN)
    tmin, bmax, tmax = times_tup

    # Create epochs object from raw and annotations
    epochs = create_epochs(raw, events, event_dict, tmin, tmax)

    # Adjust annotations for repeats
    cleaned_events = remove_dropped_events(epochs, events)

    # Get only stimulus epochs
    (
        stimulus_events,
        key_wo_pp_events,
        key_without_pp_samps_to_ms,
        key_to_pp_lag,
        pp_updown_duration,
        iti_stim_gap,
    ) = get_stimulus_epochs(epochs, event_values, event_keys, cleaned_events)
    stimulus_indices = [el[0] for el in stimulus_events]
    stim_epochs = epochs[stimulus_indices]

    # Correct missing pinpricks
    if key_wo_pp_events:
        key_wo_pp_indices = [el[0] for el in key_wo_pp_events]
        cleaned_events, stimulus_indices = correct_missing_pinpricks(
            cleaned_events, key_wo_pp_indices, stimulus_indices
        )

    # Get stimulus labels and pain ratings
    df = load_pain_ratings(data_path, sub_id)
    ground_truth_labels, pain_ratings = get_labels_and_ratings(df)

    # Correct extra/missing/incorrect key presses and
    # verify number of trials match remaining stimulus labels and pain ratings
    stimulus_labels = [el[1] for el in stimulus_events]
    cleaned_stim_labels = get_stimulus_labels(stimulus_labels, event_values)
    (
        corrected_stimulus_labels,
        corrected_ground_truth_labels,
        corrected_stim_epochs,
        corrected_pain_ratings,
    ) = verify_lists_match(
        cleaned_stim_labels,
        ground_truth_labels,
        stim_epochs,
        pain_ratings,
    )

    # Reject noisy epochs
    ar = AutoReject(random_state=RANDOM_STATE)
    _, reject_log = ar.fit_transform(stim_epochs, return_log=True)
    bad_epochs = reject_log.bad_epochs.tolist()
    dropped_epochs = [idx for idx, is_bad in enumerate(bad_epochs) if is_bad]

    # Remove dropped epochs
    stim_epochs.drop(dropped_epochs)
    ground_truth_labels = np.delete(
        corrected_ground_truth_labels, dropped_epochs, axis=0
    )
    stimulus_labels = np.delete(corrected_stimulus_labels, dropped_epochs, axis=0)
    pain_ratings = np.delete(corrected_pain_ratings, dropped_epochs, axis=0)

    # Print length of each
    print(
        f"Length of stimulus epochs: {len(stim_epochs)}\n"
        f"Length of ground truth labels: {len(ground_truth_labels)}\n"
        f"Length of pain ratings: {len(pain_ratings)}"
    )

    # Z-score stimulus epochs
    zscored_epochs = zscore_epochs(corrected_stim_epochs)

    return (zscored_epochs, stimulus_labels, pain_ratings)

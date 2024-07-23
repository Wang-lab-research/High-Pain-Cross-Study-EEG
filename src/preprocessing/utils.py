import os
import numpy as np
import mne
import pandas as pd
import pickle
import h5py
import csv
from mne.preprocessing import ICA
from pyprep.find_noisy_channels import NoisyChannels
from IPython import display
from glob import glob
from src.utils.config import Config
import src.configs.config as configs

config = Config.from_json(configs.CFGLog)

RESAMPLE_FREQ = config.parameters.sfreq
RANDOM_STATE = config.parameters.random_seed


def clear_output():
    """Clear output of the IPython display."""
    display.clear_output(wait=True)


def set_custom_montage(mne_object, montage_path):
    custom_montage = mne.channels.read_custom_montage(montage_path)
    mne_object.set_montage(custom_montage, on_missing="ignore")


def pickle_data(save_path, fname, data):
    with open(os.path.join(save_path, fname), "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {fname} to {save_path}.")


def unpickle_data(path, fname):
    with open(os.path.join(path, fname), "rb") as f:
        deserialized_object = pickle.load(f)
    return deserialized_object


def load_file(subject_id, data_path, extension="hdf5"):
    """
    Loading hdf5 file from parent data folder given sub id
    """
    for folder in os.listdir(data_path):
        if subject_id in folder:
            subject_id = folder
            break
    if extension == "hdf5":
        h5_files = glob(os.path.join(data_path, folder, f"*.{extension}"))
        return h5py.File(h5_files[0], "r")
    elif extension == "edf":
        edf_files = glob(os.path.join(data_path, folder, "*.edf")) + glob(
            os.path.join(data_path, folder, "*.EDF")
        )
        return mne.io.read_raw_edf(edf_files[0], preload=True)
    elif extension == "csv":
        csv_files = glob(os.path.join(data_path, folder, "*.csv"))
        return pd.read_csv(csv_files[0])


def get_time_window(peri_stim_time_win=None):
    """
    Get the tmin,tmax,bmax for any custom time window.
    Also get the custom save path.
    """
    bmax = 0.0
    if peri_stim_time_win is None:
        t_win = float(
            input(
                "Please enter the peri-stimulus time window."
                + "\nEx: '0 (default)' = [-0.2,0.8], '2' = [-1.0,1.0], etc...\n\n>> "
            )
        )
    else:
        t_win = float(peri_stim_time_win)

    if t_win == 0.0:
        tmin, tmax = -0.2, 0.8
        time_win_path = ""
    else:
        tmin, tmax = -t_win / 2, t_win / 2
    print(f"[{tmin},{bmax},{tmax}]")
    time_win_path = f"{int(t_win)}_sec_time_window/"
    # print(time_win_path)
    return (tmin, bmax, tmax), time_win_path


def make_sub_time_win_path(
    subject_id, save_path_cont, save_path_zepo, include_zepochs=True
):
    """
    Make a subject's time window data path
    """
    subpath_cont = os.path.join(save_path_cont, subject_id)
    if not os.path.exists(subpath_cont):  # continuous
        os.mkdir(subpath_cont)
    if include_zepochs:
        subpath_zepo = os.path.join(save_path_zepo, subject_id)
        if not os.path.exists(subpath_zepo):  # zepochs
            os.mkdir(subpath_zepo)
    return subpath_cont, subpath_zepo


def get_raw_path(subject_id: str, data_dir: str) -> tuple:
    """
    Find and return the path to the EDF data file for the given subject ID.

    Args:
        subject_id: The subject ID.
        data_dir: The directory where the data files are stored.

    Returns:
        tuple: A tuple containing the subject folder path and the path to the EDF data file.

    Raises:
        ValueError: If the subject ID is not found in the data path or if more than one EDF file is found.
    """
    subject_folder = next(
        (folder for folder in os.listdir(data_dir) if folder.startswith(subject_id)),
        None,
    )
    if subject_folder is None:
        raise ValueError(f"Subject ID {subject_id} not found in {data_dir}.")

    subject_folder_path = os.path.join(data_dir, subject_folder)
    edf_files = glob(os.path.join(subject_folder_path, "*.EDF")) + glob(
        os.path.join(subject_folder_path, "*.edf")
    )
    if len(edf_files) != 1:
        raise ValueError(
            f"Expected one EDF file in {subject_folder_path}, found {len(edf_files)}"
        )

    return subject_folder, edf_files[0]


def create_resting_csv(data_path, save_path, subject_id, annotation_keys):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_list = os.listdir(data_path)
    raw_edf_files = [file for file in file_list if file.lower().endswith(".edf")]
    file = raw_edf_files[0]
    file_path = os.path.join(data_path, file)

    raw = mne.io.read_raw_edf(file_path, preload=False)

    events_from_annot, event_dict = mne.events_from_annotations(
        raw, event_id=annotation_keys
    )
    # Extract timestamps from the events
    timestamps_with_id = [
        (event[0] / raw.info["sfreq"], event[2]) for event in events_from_annot
    ]

    saved_times = [None, None, None, None]  # Initialize an array to store timestamps

    for idx, (timestamp, event_id) in enumerate(timestamps_with_id):
        # Find first instance of event ID 1, then the first instance of event ID 9 afterwards
        # Find first instance of event ID 2, the the first instance of event ID 9 afterwards
        description = raw.annotations[idx]["description"].lower()
        if event_id == 1 or "closed" in description:
            if saved_times[0] is None:  # Check if the first index is empty
                saved_times[0] = timestamp
        elif (
            (event_id == 9 or "end" in description)
            and idx > 0
            and (
                timestamps_with_id[idx - 1][1] == 1
                or "closed" in raw.annotations[idx - 1]["description"].lower()
            )
        ):
            saved_times[1] = timestamp
        elif event_id == 2 or "open" in description:
            if saved_times[2] is None:  # Check if the third index is empty
                saved_times[2] = timestamp
        elif (
            (event_id == 9 or "end" in description)
            and idx > 0
            and (
                timestamps_with_id[idx - 1][1] == 2
                or "open" in raw.annotations[idx - 1]["description"].lower()
            )
        ):
            saved_times[3] = timestamp

    print("Saved Times:", saved_times)

    csv_file_name = f"{subject_id}_RestingTStamps.csv"
    # Construct the full file path
    full_file_path = os.path.join(save_path, csv_file_name)

    # Write data to the CSV file
    with open(full_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Seconds"])

        for timestamp in saved_times:
            writer.writerow([timestamp])
    return full_file_path


def load_csv(subject_id, csv_path):
    """
    Function purpose: Obtain the CSV file with timestamps for resting EEG timeframe
    Inputs: subject_id = subject id of interest
            csv_path = file path to the folder with the csv
    Outputs: Corresponding csv file for subject of interest
    """
    csv_folder = os.listdir(csv_path)
    for file in csv_folder:
        if file.endswith(".csv") and subject_id in file:
            return pd.read_csv(os.path.join(csv_path, file))
    print(f"CSV file with {subject_id} not found in the folder.")
    return None


def crop_by_resting_times(raw, start, stop, subject_id, save_path, category):
    """
    Function purpose: Create cropped files and save them.
    Inputs: raw = *raw.fif file, start = beginning timepoint in seconds, stop = ending timepoint in seconds
            save_path = file path to file for saved cropped data
            category = name for file (eyes_closed, noise, eyes_open)
    Outputs: cropped file in *raw.fif format
    """
    filename = f"{subject_id}_{category}-raw.fif"
    filepath = os.path.join(save_path, filename)
    cropped = raw.copy().crop(tmin=start, tmax=stop)
    cropped.save(filepath, overwrite=True)
    return cropped


def get_cropped_resting_EEGs(subject_id, raw, csv_path, save_path, include_noise=True):
    """
    Function purpose: Create recording of the full resting EEG
    Inputs: subject_id = subject ID ie the patient number,
            raw = *{subject_id}...raw.fif file
            csv_path = file path for the folder with the csv with the resting timestamps
            save_path = file path for saving the recording
    Outputs: *raw.fif file with recording for eyes closed only (e.g. 007_eyes_closed-raw.fif)
            *raw.fif file with recording for noise calibration only (e.g. 007_noise-raw.fif)
            *raw.fif file with recording for eyes open only (e.g. 007_eyes_open-raw.fif)
    """
    timestamp_csv = load_file(subject_id, csv_path, extension="csv")
    if timestamp_csv is None:
        print(f"No CSV for {subject_id} found, no cropped recordings created")
        return None

    print(f"Loading CSV for {subject_id}")

    EC_start, EC_stop, EO_start, EO_stop = timestamp_csv["Seconds"][0:4]

    # Establish timestamps assuming enough recorded for 5 mins of eyes open noise = 2 mins, EO = 3 mins
    # Case 1: Normal case, EO is at least 5 mins long
    noise_start = EO_start
    noise_stop = noise_start + 120
    cropped_EO_start = noise_stop  # Need to reset below

    EO_duration = EO_stop - EO_start

    # Adjust durations based on the length of the recording
    if EO_duration >= 300:  # Resting recording is at least 5 mins
        noise_stop = EO_start + 120
    elif EO_duration >= 270:  # Resting recording is between 4.5-5 mins
        noise_stop = EO_start + 90
    else:  # Resting recording is less than 4.5 mins
        noise_stop = EO_start + 60

    # Update cropped_EO_start after adjusting noise duration
    cropped_EO_start = noise_stop

    cropped_EO_stop = cropped_EO_start + 180  # EO is 3 minutes

    # Send message if eyes closed is shorter than 3 mins, otherwise default is 3 min eyes closed recording
    if (EC_stop - EC_start) < 180:
        if include_noise:
            print(
                f"Eyes closed is not longer than 3 mins. Length of EC reading is: {EC_stop- EC_start} seconds."
            )
    else:
        EC_stop = EC_start + 180

    print(f"Cropping files for {subject_id}\n")

    # Include noise or not
    if include_noise:
        # Crop and save the cropped raw data to a raw.fif file
        EC_cropped = crop_by_resting_times(
            raw, EC_start, EC_stop, subject_id, save_path, "eyes_closed"
        )
        noise_cropped = crop_by_resting_times(
            raw, noise_start, noise_stop, subject_id, save_path, "noise"
        )
        EO_cropped = crop_by_resting_times(
            raw, cropped_EO_start, cropped_EO_stop, subject_id, save_path, "eyes_open"
        )
    else:
        # Crop and save the cropped raw data to a raw.fif file
        EO_cropped = crop_by_resting_times(
            raw, EO_start, EO_stop, subject_id, save_path, "eyes_open"
        )
        EC_cropped = None
        noise_cropped = None
    return EC_cropped, noise_cropped, EO_cropped


def remove_trailing_zeros(raw, subject_id, channel_threshold=0.5):
    """
    Removes trailing zeros from raw data channels.

    Parameters:
    - raw: The raw data object containing time-series data.
    - subject_id: Subject identifier.

    Returns:
    - raw: The potentially modified raw data object after cropping.
    """
    raw_duration = raw.times[-1]
    raw_data = raw.get_data()
    trailing_zeroes_present = False
    channels_with_trailing_zeros = 0

    for channel_data in raw_data:
        consecutive_zeros = 0
        for i in range(len(channel_data)):
            if channel_data[i] == 0.0:
                consecutive_zeros += 1
            else:
                consecutive_zeros = 0
            if consecutive_zeros >= 100:
                channels_with_trailing_zeros += 1
                break

    # Check if the number of channels with trailing zeros exceeds the threshold
    if channels_with_trailing_zeros >= channel_threshold * raw_data.shape[0]:
        trailing_zeroes_present = True

    if trailing_zeroes_present:
        start_index = len(raw_data[0]) - consecutive_zeros
        end_index = len(raw_data[0])
        trailing_zeros_duration = (end_index - start_index) / raw.info["sfreq"]
        tmax = raw_duration - np.ceil(trailing_zeros_duration)
        if tmax <= 0:
            tmax = raw.times[1]  # Ensure tmax is valid
        raw = raw.crop(
            tmin=0,
            tmax=tmax,
            include_tmax=False,
        )

    return raw


def get_binary_pain_trials(
    subject_id, pain_ratings_raw, pain_thresh, processed_data_path
):
    pain_ratings = [
        1 if el > pain_thresh else 0 for i, el in enumerate(pain_ratings_raw)
    ]
    # use pain/no-pain dict for counting trial ratio
    event_ids_pain_dict = {
        "Pain": 1,
        "No Pain": 0,
    }

    # Count pain and no-pain trials
    unique, counts = np.unique(pain_ratings, return_counts=True)
    event_ids_inv = {v: k for k, v in event_ids_pain_dict.items()}
    unique_labels = np.vectorize(event_ids_inv.get)(unique)
    trial_counts_dict = dict(zip(unique_labels, counts))
    pain_trials_counts = list(trial_counts_dict.values())

    # If no painful trials or not enough, take note of subject_id
    if (
        len(pain_trials_counts) == 1
        or np.all([el >= 4 for el in pain_trials_counts]) is False
    ):
        # save record of which subjects don't meet the requirement
        with open(
            processed_data_path / "Insufficient_Pain_Trials_subject_ids.txt", "a"
        ) as txt_file:
            txt_file.write(subject_id / "\n")

        # set pain ratings to None
        pain_ratings = None

    return pain_ratings


def preprocess_entire(raw, subject_id):
    # raw = remove_trailing_zeros(raw, subject_id, channel_threshold=0.5)

    if "X" in raw.ch_names and len(raw.ch_names) < 64:
        rename_and_set_channel_types_32(raw)
        drop_unused_channels_32(raw)
        set_custom_montage(raw, config.parameters["32_channel_montage_file_path"])
        raise ValueError("32 channel data detected")
    else:
        handle_64_channel_case(raw, subject_id)

    apply_bandpass_filter(raw)
    apply_notch_filter(raw)
    set_average_reference(raw)
    ica = fit_ica(raw, subject_id)
    ica = find_eog_artifacts(raw, ica, subject_id)
    ica.apply(raw)
    find_and_interpolate_bad_channels(raw)
    resample_data(raw)
    inspect_data(raw, subject_id)

    return raw


def rename_and_set_channel_types_32(raw):
    raw.rename_channels({"Fp1": "EOG1"})
    raw.set_channel_types({"EOG1": "eog"})


def drop_unused_channels_32(raw):
    non_eeg_chs = ["X", "Y", "Z"] if "X" in raw.ch_names else []
    non_eeg_chs += ["Oth4"] if "Oth4" in raw.ch_names else []
    raw.drop_channels(non_eeg_chs)


def handle_64_channel_case(raw, subject_id):
    """
    Handle the case of 64 channel EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    subject_id : str
        The subject ID.
    """
    # Check if the data is from the 64 channel EEG cap
    if {"VEO"}.issubset(set(raw.ch_names)) or {"VEOG"}.issubset(set(raw.ch_names)):
        # Rename channels
        raw.rename_channels({"VEO" if "VEO" in raw.ch_names else "VEOG": "EOG1"})
        raw.rename_channels({"HEO" if "HEO" in raw.ch_names else "HEOG": "EOG2"})
        raw.set_channel_types({"EOG1": "eog", "EOG2": "eog"})

        # Drop non-EEG channels
        non_eeg_chs = ["EKG", "EMG", "Trigger"]
        raw.drop_channels(non_eeg_chs)

        # Drop extra channels
        if {"FT7", "PO5"}.issubset(set(raw.ch_names)):
            raw.drop_channels(["FT7", "FT8", "PO5", "PO6"])

        # Set montage
        set_custom_montage(raw, config.parameters.get("64_channel_montage_file_path"))

    # Check if the data is from the gTec cap
    elif "AF8" in raw.ch_names:
        # Rename channels
        raw.rename_channels({"65": "EOG1", "66": "EOG2"})
        raw.set_channel_types({"EOG1": "eog", "EOG2": "eog"})

        # Drop numeric channels
        raw.drop_channels([ch for ch in raw.ch_names if ch.isnumeric()])

        # Rename FP channels
        for i, ch in enumerate(raw.info["ch_names"]):
            if "FP" in ch:
                raw.rename_channels({ch: "Fp" + ch[2:]})

        # Create new montage
        montage_1020 = mne.channels.make_standard_montage("standard_1020")
        kept_channels = raw.info["ch_names"][:64]
        ind = [
            i
            for (i, channel) in enumerate(montage_1020.ch_names)
            if channel.lower() in map(str.lower, kept_channels)
        ]
        montage_1020_new = montage_1020.copy()
        montage_1020_new.ch_names = [montage_1020.ch_names[x] for x in ind]
        kept_channel_info = [montage_1020.dig[x + 3] for x in ind]

        montage_1020_new.dig = montage_1020.dig[0:3] + kept_channel_info

        # Drop extra channels
        if "A1" in raw.ch_names:
            raw.drop_channels(["A1", "A2"])

        # Set montage
        raw.set_montage(montage_1020_new)
    else:
        raise ValueError(f"Could not determine montage/cap for subject {subject_id}")

    return raw


def apply_notch_filter(raw):
    raw.notch_filter(60.0, notch_widths=3)


def apply_bandpass_filter(raw):
    raw.filter(l_freq=1.0, h_freq=100.0)


def resample_data(raw):
    raw.resample(RESAMPLE_FREQ, npad="auto")


def find_and_interpolate_bad_channels(raw):
    raw_pyprep = NoisyChannels(raw, random_state=RANDOM_STATE)
    raw_pyprep.find_all_bads(ransac=False, channel_wise=False, max_chunk_size=None)
    raw.info["bads"] = raw_pyprep.get_bads()
    print(f"Bad channels: {raw.info['bads']}")
    raw.interpolate_bads()


def set_average_reference(raw):
    raw.set_eeg_reference(ref_channels="average", 
                          projection=True)


def fit_ica(raw, subject_id):
    num_goods = len(raw.ch_names) - len(raw.info["bads"]) - 1  # adjust for EOG
    ica = ICA(
        n_components=int(np.floor(num_goods / 2)),
        random_state=RANDOM_STATE,
        max_iter="auto",
    )
    ica.fit(raw)
    return ica


def find_eog_artifacts(raw, ica, subject_id):
    if "EOG1" in raw.ch_names:
        print(f"{subject_id}\nfinding EOG artifacts...")
        try:
            eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=True)
            ica.exclude = eog_indices
        except ValueError:
            ica.exclude = [0, 1]
    return ica


def inspect_data(raw, subject_id):
    print(f"Subject: {subject_id}")
    print("Number of remaining channels: ", len(raw.ch_names) - len(raw.info["bads"]))
    print("Dropped channels: ", raw.info["bads"])


def crop_resting_EO(
    raw, sub_id, data_path, processed_data_path, events=None, event_ids=None
):
    # Get converted EDF file for events
    raw_edf = load_file(sub_id, data_path, extension="edf")
    sfreq = raw.info["sfreq"]

    if events is None and event_ids is None:
        events, event_ids = mne.events_from_annotations(raw_edf)

    # For now get just the events we are interested in, eyes open and stop

    # Set events indicating start and end of resting eyes open
    eyes_open_id = event_ids["KB-Marker-0 (Eyes open) "]
    stop_id = event_ids["KB-Marker-9 (END/STOP test) "]

    # Check for eyes open marker and stop marker either right after or two after the event, to accout for mistakes
    max_time = 320 * sfreq  # maximum time between eyes open and stop
    for i in range(len(events)):
        # local events
        if events[i][2] == eyes_open_id:
            this_event = events[i]
            next_event = events[i + 1]
            following_event = events[i + 2] if len(events) > i + 2 else None

            # get event ids
            this_event_id = this_event[2]
            next_event_id = next_event[2]
            following_event_id = (
                following_event[2] if following_event is not None else None
            )

            # get event times
            this_event_samples = this_event[0]
            next_event_samples = next_event[0]
            following_event_samples = (
                following_event[0] if following_event is not None else None
            )

            # save eyes open times if valid
            if (
                this_event_id == eyes_open_id
                and next_event_id == stop_id
                and (next_event_samples - this_event_samples) < max_time
            ):
                print("\nEyes open found and next event STOP")
                eyes_open_events = [i, i + 1]
                eyes_open_times = [el[0] for el in events[eyes_open_events]]
            elif (
                this_event_id == eyes_open_id
                and following_event_id == stop_id
                and (following_event_samples - this_event_samples) < max_time
            ):
                print("\nEyes open found and FOLLOWING event STOP")
                eyes_open_events = [i, i + 2]
                eyes_open_times = [el[0] for el in events[eyes_open_events]]
            elif this_event_id == eyes_open_id and (
                next_event_id != stop_id or following_event_id != stop_id
            ):
                print("\nEyes open found but NO STOP found")
                eyes_open_events = [
                    i,
                ]
                eyes_open_times = [el[0] for el in events[eyes_open_events]]
                eyes_open_times.append(eyes_open_times[0] + 300 * sfreq)
            else:
                raise ValueError("\nError, an eyes open window cannot be created")

            # Get eyes open times
            eyes_open_times_seconds = [el / sfreq for el in eyes_open_times]
            break

    print(f"\nEyes open times: {eyes_open_times_seconds}")

    # save cropped data
    raw.crop(tmin=eyes_open_times_seconds[0], tmax=eyes_open_times_seconds[-1])

    raw.save(processed_data_path / f"{sub_id}_eyes_open-raw.fif", overwrite=True)

    # crop data
    return raw


def snip_span(raw, t1, t2):
    """
    Extracts the data from raw as numpy array, snip a middle section out based on two time values,
    reconcatenate the numpy array, then create an mne.RawArray object using the raw.info

    Args:
    raw (mne.io.Raw): Input raw data
    t1 (float): Start time of the section to be removed in seconds
    t2 (float): End time of the section to be removed in seconds

    Returns:
    mne.io.RawArray: Processed raw data
    """

    # Extract the data and times from raw
    data, times = raw[:]

    # Convert times to indices
    idx1 = np.argmin(np.abs(times - t1))
    idx2 = np.argmin(np.abs(times - t2))

    # Snip out the middle section and reconcatenate
    processed_data = np.concatenate((data[:, :idx1], data[:, idx2:]), axis=1)

    # Create a new MNE RawArray object with the processed data
    processed_raw = mne.io.RawArray(processed_data, raw.info)

    return processed_raw

import os
import numpy as np
import mne
import pandas as pd
import pickle
import h5py
from mne.preprocessing import ICA
from pyprep.find_noisy_channels import NoisyChannels
from IPython import display
from glob import glob
from src.configs.config import CFGLog

RESAMPLE_FREQ = CFGLog["parameters"]["sfreq"]
RANDOM_STATE = CFGLog["parameters"]["random_seed"]


def clear_display():
    display.clear_output(wait=True)


def load_raw_data(data_path, sub_id, eog):
    """
    Load raw EDF data with specified EOG channel.
    """
    sub_folder = next(
        sub_folder
        for sub_folder in os.listdir(os.path.join(data_path))
        if (sub_folder.startswith(sub_id))
    )
    eeg_data_raw_file = os.path.join(
        data_path,
        sub_folder,
        next(
            subfile
            for subfile in os.listdir(os.path.join(data_path, sub_folder))
            if (subfile.endswith((".edf", ".EDF")))
        ),
    )

    return mne.io.read_raw_edf(eeg_data_raw_file, eog=[eog], preload=True)


def set_montage(mne_obj, montage):
    """
    Set custom montage for Raw or Epochs object.
    """
    print("setting custom montage...")
    print(montage)
    if isinstance(montage, str):
        relative_path = os.path.join(os.path.dirname(__file__), montage)
        dig_montage = mne.channels.read_custom_montage(relative_path)
        mne_obj.set_montage(dig_montage, on_missing="ignore")
    else:
        mne_obj.set_montage(montage, on_missing="ignore")


# functions for serialization
def pickle_data(save_path, fname, data):
    with open(os.path.join(save_path, fname), "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {fname} to {save_path}.")


def unpickle_data(path, fname):
    with open(os.path.join(path, fname), "rb") as f:
        deserialized_object = pickle.load(f)
    return deserialized_object


def load_file(sub_id, data_path, extension="hdf5"):
    """
    Loading hdf5 file from parent data folder given sub id
    """
    for folder in os.listdir(data_path):
        if sub_id in folder:
            sub_id = folder
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
    sub_id, save_path_cont, save_path_zepo, include_zepochs=True
):
    """
    Make a subject's time window data path
    """
    subpath_cont = os.path.join(save_path_cont, sub_id)
    if not os.path.exists(subpath_cont):  # continuous
        os.mkdir(subpath_cont)
    if include_zepochs:
        subpath_zepo = os.path.join(save_path_zepo, sub_id)
        if not os.path.exists(subpath_zepo):  # zepochs
            os.mkdir(subpath_zepo)
    return subpath_cont, subpath_zepo


def get_raw_data_file_path(subject_id, data_path):
    """
    Find and return the path to the EDF data file for the given subject ID.

    Args:
        subject_id (str): The subject ID.
        data_path (str): The directory where the data files are stored.

    Returns:
        str: The path to the EDF data file for the given subject ID.
    """
    subject_folder = next(
        (folder for folder in os.listdir(data_path) if subject_id in folder),
        None,
    )
    if subject_folder is None:
        raise ValueError(f"Subject ID {subject_id} not found in {data_path}.")
    subject_folder = os.path.join(data_path, subject_folder)
    data_files = []
    data_files += glob(subject_folder + "/*.EDF")
    if len(data_files) != 1:
        raise ValueError(
            f"Expected one EDF file in {subject_folder}, found {len(data_files)}"
        )
    return data_files[0]


def crop_by_resting_times(raw, start, stop, sub_id, save_path, category):
    """
    Function purpose: Create cropped files and save them.
    Inputs: raw = *raw.fif file, start = beginning timepoint in seconds, stop = ending timepoint in seconds
            save_path = file path to file for saved cropped data
            category = name for file (eyes_closed, noise, eyes_open)
    Outputs: cropped file in *raw.fif format
    """
    filename = f"{sub_id}_{category}-raw.fif"
    filepath = os.path.join(save_path, filename)
    cropped = raw.copy().crop(tmin=start, tmax=stop)
    cropped.save(filepath, overwrite=True)
    return cropped


def get_cropped_resting_EEGs(sub_id, raw, csv_path, save_path, include_noise=True):
    """
    Function purpose: Create recording of the full resting EEG
    Inputs: sub_id = subject ID ie the patient number,
            raw = *{sub_id}...raw.fif file
            csv_path = file path for the folder with the csv with the resting timestamps
            save_path = file path for saving the recording
    Outputs: *raw.fif file with recording for eyes closed only (e.g. 007_eyes_closed-raw.fif)
            *raw.fif file with recording for noise calibration only (e.g. 007_noise-raw.fif)
            *raw.fif file with recording for eyes open only (e.g. 007_eyes_open-raw.fif)
    """
    timestamp_csv = load_file(sub_id, csv_path, extension="csv")
    if timestamp_csv is None:
        print(f"No CSV for {sub_id} found, no cropped recordings created")
        return None

    print(f"Loading CSV for {sub_id}")

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

    print(f"Cropping files for {sub_id}\n")

    # Include noise or not
    if include_noise:
        # Crop and save the cropped raw data to a raw.fif file
        EC_cropped = crop_by_resting_times(
            raw, EC_start, EC_stop, sub_id, save_path, "eyes_closed"
        )
        noise_cropped = crop_by_resting_times(
            raw, noise_start, noise_stop, sub_id, save_path, "noise"
        )
        EO_cropped = crop_by_resting_times(
            raw, cropped_EO_start, cropped_EO_stop, sub_id, save_path, "eyes_open"
        )
    else:
        # Crop and save the cropped raw data to a raw.fif file
        EO_cropped = crop_by_resting_times(
            raw, EO_start, EO_stop, sub_id, save_path, "eyes_open"
        )
        EC_cropped = None
        noise_cropped = None
    return EC_cropped, noise_cropped, EO_cropped


def remove_trailing_zeros(raw, sub_id, sfreq):
    """
    Removes trailing zeros from raw data channels.

    Parameters:
    - raw: The raw data object containing time-series data.
    - sub_id: Subject identifier.
    - sfreq: Sampling frequency.

    Returns:
    - raw: The potentially modified raw data object after cropping.
    - need_crop: A boolean indicating if cropping was performed.
    """
    raw_dur = raw.times[-1]
    raw_data = raw.get_data()
    need_crop = False

    print(f"Looking for trailing zeros in subject {sub_id}")

    zero_count = 0
    ch = raw_data[0]
    for i in range(len(ch)):
        if ch[i] == 0.0:
            zero_count += 1
            if zero_count >= 100:
                start_index = i - (zero_count - 1)
                end_index = len(ch)
                print(
                    f"{zero_count} consecutive zeros found starting at index {start_index}"
                )
                zeros_dur = (end_index - start_index) / sfreq
                print(f"Duration: {zeros_dur} sec")
                need_crop = True
                break
        else:
            zero_count = 0
    if need_crop:
        print("Need to crop trailing zeros")
        raw = raw.crop(tmin=0, tmax=raw_dur - np.ceil(zeros_dur), include_tmax=False)

    return raw, need_crop


def get_binary_pain_trials(sub_id, pain_ratings_raw, pain_thresh, processed_data_path):
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

    # If no painful trials or not enough, take note of sub_id
    if (
        len(pain_trials_counts) == 1
        or np.all([el >= 4 for el in pain_trials_counts]) is False
    ):
        # save record of which subjects don't meet the requirement
        with open(
            processed_data_path / "Insufficient_Pain_Trials_Sub_IDs.txt", "a"
        ) as txt_file:
            txt_file.write(sub_id / "\n")

        # set pain ratings to None
        pain_ratings = None

    return pain_ratings


def to_raw(data_path, sub_id, save_path, csv_path, include_noise):
    """
    Preprocess raw EDF data to filtered FIF format.
    """
    for sub_folder in os.listdir(data_path):
        if sub_folder.startswith(sub_id):
            save_fname_fif = sub_id + "_preprocessed-raw.fif"
            print(sub_id, save_fname_fif)
            break

    # read data, set EOG channel, and drop unused channels
    print(f"{sub_id}\nreading raw file...")
    raw = load_raw_data(data_path, sub_folder, "EOG")
    sfreq = raw.info["sfreq"]
    # Assuming `raw`, `sub_id`, and `raw_sfreq` are already defined:
    raw_cropped, was_cropped = remove_trailing_zeros(raw, sub_id, sfreq)
    if was_cropped:
        print("Data was cropped to remove trailing zeros.")
        raw = raw_cropped

    # if channel names are numeric, drop them
    raw.drop_channels([ch for ch in raw.ch_names if ch.isnumeric()])

    # read data, set EOG channel, and drop unused channels
    montage_fname = "../montages/Hydro_Neo_Net_64_xyz_cms_No_FID.sfp"
    Fp1_eog_flag = 0
    # 32 channel case
    if "X" in raw.ch_names and len(raw.ch_names) < 64:
        raw = load_raw_data(data_path, sub_folder, "Fp1")

        # replace with EOG
        raw.rename_channels({"Fp1": "EOG"})

        Fp1_eog_flag = 1

        non_eeg_chs = ["X", "Y", "Z"] if "X" in raw.ch_names else []
        non_eeg_chs += ["Oth4"] if "Oth4" in raw.ch_names else []

        raw.drop_channels(non_eeg_chs)
        montage_fname = "../montages/Hydro_Neo_Net_32_xyz_cms_No_Fp1.sfp"
        set_montage(raw, montage_fname)

    # 64 channel case
    else:
        # For Compumedics 64 channel cap
        if "VEO" in raw.ch_names or "VEOG" in raw.ch_names:
            # eog_adj = 5
            raw = load_raw_data(
                data_path, sub_folder, "VEO" if "VEO" in raw.ch_names else "VEOG"
            )
            # replace VEO with EOG
            raw.rename_channels({"VEO" if "VEO" in raw.ch_names else "VEOG": "EOG"})

            non_eeg_chs = (
                ["HEOG", "EKG", "EMG", "Trigger"]
                if "HEOG" in raw.ch_names
                else ["HEO", "EKG", "EMG", "Trigger"]
            )
            raw.drop_channels(non_eeg_chs)
            montage_fname = "../montages/Hydro_Neo_Net_64_xyz_cms_No_FID.sfp"
            set_montage(raw, montage_fname)

            # For subjects C24, 055, 056, 047 the wrong montage was used
            if {"FT7", "PO5"}.issubset(set(raw.ch_names)):
                raw.drop_channels(["FT7", "FT8", "PO5", "PO6"])
                montage_fname = "../montages/Hydro_Neo_Net_64_xyz_cms_No_FID_Caps.sfp"
                set_montage(raw, montage_fname)
        if "EEG66" in raw.ch_names:
            non_eeg_chs = ["EEG66", "EEG67", "EEG68", "EEG69"]
            raw.drop_channels(non_eeg_chs)

        # For 64 channel gTec cap
        if "AF8" in raw.ch_names:
            # Form the 10-20 montage
            mont1020 = mne.channels.make_standard_montage("standard_1020")

            # Rename capitalized channels to lowercase
            print("Renaming capitalized channels to lowercase...")
            for i, ch in enumerate(raw.info["ch_names"]):
                if "FP" in ch:
                    raw.rename_channels({ch: "Fp" + ch[2:]})

            # Choose what channels you want to keep
            # Make sure that these channels exist e.g. T1 does not exist in the standard 10-20 EEG system!
            kept_channels = raw.info["ch_names"][:64]
            ind = [
                i
                for (i, channel) in enumerate(mont1020.ch_names)
                if channel.lower() in map(str.lower, kept_channels)
            ]
            mont1020_new = mont1020.copy()
            # Keep only the desired channels
            mont1020_new.ch_names = [mont1020.ch_names[x] for x in ind]
            kept_channel_info = [mont1020.dig[x + 3] for x in ind]
            # Keep the first three rows as they are the fiducial points information
            mont1020_new.dig = mont1020.dig[0:3] + kept_channel_info
            set_montage(raw, mont1020_new)

    # # 007 and 010 had extremely noisy data near the ends of their recordings.
    # # Crop it out.
    # if sub_id == "007":
    #     raw = raw.crop(tmax=1483)
    # elif sub_id == "010":
    #     raw.crop(tmax=1997.8)

    # high level inspection
    print(raw.ch_names)
    print(len(raw.ch_names))

    # apply notch filter
    print(f"{sub_id}\napplying notch filter...")
    raw = raw.notch_filter(60.0, notch_widths=3)
    clear_display()

    # apply bandpass filter
    print(f"{sub_id}\napplying bandpass filter...")
    raw = raw.filter(l_freq=1.0, h_freq=100.0)
    clear_display()

    # resample data to decrease file size
    print(
        f"{sub_id}\nresampling data from {raw.info['sfreq']} Hz to {RESAMPLE_FREQ} Hz..."
    )
    raw.resample(RESAMPLE_FREQ, npad="auto")
    clear_display()

    # find bad channels automatically
    print(f"{sub_id}\nremoving bad channels...")
    raw_pyprep = NoisyChannels(raw, random_state=RANDOM_STATE)
    raw_pyprep.find_all_bads(ransac=False, channel_wise=False, max_chunk_size=None)
    raw.info["bads"] = raw_pyprep.get_bads()
    print(f"{sub_id} bad channels: {raw.info['bads']}")
    raw.interpolate_bads()
    # clear_display()

    # re-reference channels
    print(f"{sub_id}\nre-referencing channels to average...")
    raw, _ = mne.set_eeg_reference(raw, ref_channels="average", copy=True)
    # clear_display()

    # Drop reference channels
    if "A1" in raw.ch_names:
        raw.drop_channels(["A1", "A2"])

    # fit ICA
    print(f"{sub_id}\nfitting ICA...")
    num_goods = len(raw.ch_names) - len(raw.info["bads"]) - 1  # adjust for EOG
    ica = ICA(
        n_components=int(np.floor(num_goods / 2)),
        random_state=RANDOM_STATE,
        max_iter="auto",
    )
    ica.fit(raw)
    # clear_display()

    # find EOG artifacts
    print(raw.ch_names)
    if "EOG" in raw.ch_names:
        print(f"{sub_id}\nfinding EOG artifacts...")
        try:
            eog_indices, eog_scores = ica.find_bads_eog(raw, threshold="auto")
            ica.exclude = eog_indices
        except ValueError:
            ica.exclude = [0, 1]
        # clear_display()

    # apply ICA
    print(f"{sub_id}\napplying ICA...")
    ica.apply(raw)
    # clear_display()

    # save copy of data
    print(f"Saving processed data as '{save_fname_fif}'...")

    if "VEO" in raw.ch_names:
        raw.drop_channels("VEO")
    elif "VEOG" in raw.ch_names:
        raw.drop_channels("VEOG")
    elif Fp1_eog_flag:
        montage_fname = "../montages/Hydro_Neo_Net_32_xyz_cms_No_Fp1.sfp"
        set_montage(raw, montage_fname)

    (
        eyes_closed_recording,
        noise_recording,
        eyes_open_recording,
    ) = get_cropped_resting_EEGs(
        sub_id, raw, csv_path, save_path, include_noise=include_noise
    )  # get_cropped_resting_EEGs saves the three resting recordings into same folder as raw

    # No need to save raw anymore, saving the cropped files instead
    # raw.save(save_path+save_fname_fif,
    #          verbose=True, overwrite=True)
    # clear_display()

    # high level inspection
    print(raw.ch_names)
    print("\nNumber of remaining channels: ", len(raw.ch_names) - len(raw.info["bads"]))
    print("\nDropped channels: ", raw.info["bads"])

    print("Raw data preprocessing complete.")

    # clear_display()

    return raw, eyes_closed_recording, noise_recording, eyes_open_recording

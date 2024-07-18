from glob import glob
import os
import pickle
import scipy.io as sio
import numpy as np
import mne
from tabulate import tabulate
from typing import Dict, List, Union
from src.preprocessing import utils as pre_utils
from src.preprocessing import utils_epo as pre_utils_epo
from src.utils.config import Config
import src.configs.config as configs

config = Config.from_json(configs.CFGLog)


class Subject:
    """
    Individual level subject class for performing preprocessing.
    Attributes:
        subject_id
        group
        raw
        raw_file_path
        preprocessed_raw
        preprocessed_data_path
        eyes_open
        epochs
        stimulus_labels
        pain_ratings
        events
        stc_eyes_open
        stc_epochs
    Methods:
        load_raw()
        preprocess()
        get_cleaned_eyes_open()
        get_cleaned_epochs()
        get_stc_eyes_open()
        get_stc_epochs()
        load_epochs()
        load_epochs_info()
        save()
        pkl_exists()
    """

    def __init__(self, subject_id: str, group: str, preprocessed_data_path: str = None):
        assert isinstance(subject_id, str), "Subject ID must be a string"
        self.subject_id = subject_id
        self.group = group

        # data paths
        data_dir = os.path.abspath(configs.CFGLog["data"][group]["path"])

        self.subject_folder, self.raw_file_path = pre_utils.get_raw_path(
            subject_id=subject_id, data_dir=data_dir
        )
        self.preprocessed_data_path = config.output.parent_save_path

    def __str__(self):
        table = [[self.subject_id, self.group]]
        headers = ["Subject ID", "Group"]
        return tabulate(table, headers=headers, tablefmt="grid")

    def load_raw(self):
        self.raw = mne.io.read_raw_edf(self.raw_file_path, preload=True)
        print(f"Loaded raw for subject {self.subject_id}")

    def load_preprocessed(self):
        preprocessed_file_path = os.path.join(
            self.preprocessed_data_path, f"{self.subject_id}_preprocessed_raw.pkl"
        )
        with open(preprocessed_file_path, "rb") as file:
            self.preprocessed_raw = pickle.load(file)
        print(f"Loaded preprocessed entire for subject {self.subject_id}")

    def load_epochs(self):
        with open(
            f"{self.preprocessed_data_path}/{self.subject_id}_epochs.pkl",
            "rb",
        ) as file:
            self.epochs = pickle.load(file)
        self.epochs.data = self.epochs.get_data(copy=True)
        print(f"Loaded epochs for subject {self.subject_id}")

    def preprocess(self, overwrite: bool = False):
        if not self.file_exists("preprocessed_raw", "pkl") or overwrite:
            self.load_raw()
            self.preprocessed_raw = pre_utils.preprocess_entire(
                self.raw, self.subject_id
            )
            self.save(self.preprocessed_raw, "preprocessed_raw")
            self.save(self.preprocessed_raw, "preprocessed_raw", as_vhdr=True, overwrite=True)
        else:
            self.load_preprocessed()

    def get_cleaned_eyes_open(self):
        if not self.pkl_exists("eyes_open"):
            pass
            self.save(self.eyes_open, "eyes_open")
        else:
            self.eyes_open = self.load("eyes_open")
        # Input: raw from self.raw
        # Steps:
        # 1. Identify eyes open time frames
        # 2. Remove any eroneous KB markers/triggers
        # 2. Crop to just eyes open
        # Output: saved .fif file with just eyes open
        #   and return just eyes_open data

    def get_cleaned_epochs(self, TIME_RANGE, PERISTIM_TIME_WIN):
        if not self.pkl_exists("epochs"):
            self.epochs, self.stimulus_labels, self.pain_ratings, self.events = (
                pre_utils_epo.preprocess_epochs(
                    self.preprocessed_raw,
                    self.subject_id,
                    self.subject_folder,
                    TIME_RANGE,
                    PERISTIM_TIME_WIN,
                )
            )
            for as_mat in [False, True]:
                self.save(self.epochs, "epochs", as_mat=as_mat)
                self.save(self.stimulus_labels, "stimulus_labels", as_mat=as_mat)
                self.save(self.pain_ratings, "pain_ratings", as_mat=as_mat)
                self.save(self.events, "events", as_mat=as_mat)
        else:
            self.epochs = self.load_epochs()
            self.load_epochs_info(self.preprocessed_data_path)

    def get_stc_eyes_open(self):
        # TODO add if not self.pkl_exists("stc_eyes_open"):
        stc_eyes_open = None
        self.save(stc_eyes_open, "stc_eyes_open")

    def get_stc_epochs(self):
        # TODO add if not self.pkl_exists("stc_epochs"):
        stc_epochs = None

        self.save(stc_epochs, "stc_epochs")

    def load_epochs_info(self):
        """Load epochs info from preprocessed data path"""
        file_paths = {
            "stimulus_labels": f"{self.subject_id}_stimulus_labels.pkl",
            "pain_ratings": f"{self.subject_id}_pain_ratings.pkl",
            "events": f"{self.subject_id}_events.pkl",
            "drop_log": f"{self.subject_id}_drop_log.pkl",
        }

        for file_name, file_path in file_paths.items():
            with open(os.path.join(self.preprocessed_data_path, file_path), "rb") as file:
                setattr(self, file_name, pickle.load(file))

    def save(
        self, data_object, object_name: str, as_mat: bool = False, as_vhdr: bool = False, 
        overwrite: bool = False
    ):
        if object_name == "stc_eyes_open":
            save_path = config.output.parent_stc_save_path.eyes_open
        elif object_name == "stc_epochs":
            save_path = config.output.parent_stc_save_path.epochs
        else:
            save_path = config.output.parent_save_path

        save_file_path = os.path.join(save_path, f"{self.subject_id}_{object_name}.pkl")

        if as_mat:
            save_file_path = save_file_path.replace(".pkl", ".mat")
            sio.savemat(save_file_path, {object_name: data_object})
        if as_vhdr:
            save_file_path = save_file_path.replace(".pkl", ".vhdr")
            data_object.export(save_file_path, overwrite=overwrite)
        else:
            with open(save_file_path, "wb") as file:
                pickle.dump(data_object, file)
        print(f"Saved {object_name} to {save_file_path}.")

    def file_exists(self, object_name: str, file_type: str = "pkl"):
        if object_name == "stc_eyes_open":
            save_path = config.output.parent_stc_save_path.eyes_open
        elif object_name == "stc_epochs":
            save_path = config.output.parent_stc_save_path.epochs
        else:
            save_path = config.output.parent_save_path

        save_file_path = os.path.join(
            save_path, f"{self.subject_id}_{object_name}.{file_type}"
        )
        return os.path.exists(save_file_path)

    def resample_events(self, original_frequency, target_frequency):
        """
        Resample event times from an original frequency to a target frequency.

        Args:
            original_frequency (float): The original frequency of the events.
            target_frequency (float): The target frequency to resample the events to.

        Updates:
            Updates the `events` attribute with resampled event times.
        """
        # Calculate the new event times by adjusting the original times to the target frequency.
        resampled_event_times = (self.events[:, 0] // original_frequency) * target_frequency

        # Reconstruct the events array with new times, maintaining other columns.
        self.events = [
            [int(resampled_event_times[i]), int(self.events[i, 1]), int(self.events[i, 2])]
            for i in range(len(resampled_event_times))
        ]
        self.events = np.array(self.events)

        # Save the resampled events in both pickle and MATLAB formats.
        self.save(self.events, "events")
        self.save(self.events, "events", as_mat=True)

    def concatenate_epochs(self, save=True, overwrite=False):
        from mne.io import concatenate_raws, RawArray
        
        if self.epochs is None: 
            self.load_epochs()

        raw_list = []
        for epoch in self.epochs:
            raw = RawArray(epoch, self.epochs.info)
            raw_list.append(raw)

        raw_concatenated = concatenate_raws(raw_list)

        self.concatenated_epochs = raw_concatenated
        
        self.save(raw_concatenated, "epochs_concatenated", as_vhdr=True, overwrite=overwrite)
        
    def reject_and_update_epochs(self):
        dropped_epochs = pre_utils_epo.reject_bad_epochs(self.epochs)

        # Update epochs info (stimulus labels, pain ratings, events, drop_log)
        self.epochs.drop(dropped_epochs)
        self.stimulus_labels = np.delete(self.stimulus_labels, dropped_epochs, axis=0)
        self.pain_ratings = np.delete(self.pain_ratings, dropped_epochs, axis=0)
        self.events = np.delete(self.events, dropped_epochs, axis=0)
        
        # Save updated info
        self.save(self.stimulus_labels, "stimulus_labels")
        self.save(self.pain_ratings, "pain_ratings")
        self.save(self.events, "events")
                

class Group:
    def __init__(self, subjects: List[Subject]):
        assert isinstance(subjects, list), "Input must be a list"
        assert all(
            [isinstance(el, Subject) for el in subjects]
        ), "Input must be a list of Subjects"
        self.subjects = subjects
        self.group = subjects[0].group

    def __str__(self):
        return f"Subjects: {self.subjects}"


class SubjectProcessor:
    def __init__(self, paths_dict: Dict[str, str], roi_acronyms: List[str]):
        self.yes_list = []
        self.no_list = []
        self.maybe_list = []

        # Define paths and settings
        self.paths_dict = paths_dict
        self.processed_data_path = self.paths_dict["processed_data_path"]
        self.stc_path = self.paths_dict["stc_path"]
        self.EO_eyes_open_data_path = self.paths_dict["EO_eyes_open_data_path"]
        self.zscored_epochs_data_path = self.paths_dict["zscored_epochs_data_path"]

        self.sfreq = 400  # Hz
        self.roi_acronyms = roi_acronyms

    def _fill_nan_channels(self, epochs):
        incomplete_ch_names = epochs.info["ch_names"]
        complete_ch_names = config.parameters.ch_names
        complete_ch_names = [ch_name.upper() for ch_name in complete_ch_names]
        missing_ch_ids = [
            i
            for i in range(len(complete_ch_names))
            if complete_ch_names[i] not in incomplete_ch_names
        ]

        data = epochs.get_data(copy=False)
        data = np.insert(data, missing_ch_ids, np.nan, axis=1)

        info = mne.create_info(
            ch_names=complete_ch_names, sfreq=self.sfreq, ch_types="eeg"
        )

        epochs = mne.EpochsArray(data, info)
        return epochs

    def _load_epochs(self, subject_id: str):
        print(f"\nLoading Epochs for {subject_id}...")
        epo_fname = glob(f"{self.processed_data_path}/{subject_id}*epo.fif")[0]
        epochs = mne.read_epochs(epo_fname)
        assert isinstance(
            epochs, mne.epochs.EpochsFIF
        ), "Input must be an Epochs object"

        if len(epochs.info["ch_names"]) < 64:
            epochs = self._fill_nan_channels(epochs)
        epochs = np.nanmean(epochs.get_data(copy=False), axis=0)
        sem = np.nanstd(epochs.get_data(copy=False), axis=0) / np.sqrt(len(epochs))
        return epochs, epochs, sem

    def _load_stc_epochs(self, subject_id: str):
        print(f"Loading STC epochs for {subject_id}...")
        stc_epo_fname = glob(
            f"{self.zscored_epochs_data_path}/{subject_id}_epochs.pkl"
        )[0]
        stc_epo = pickle.load(open(stc_epo_fname, "rb"))
        stc_epo = np.array(stc_epo)

        stim_fname = glob(f"{self.processed_data_path}/{subject_id}*stim_labels.mat")[0]
        stim_labels = sio.loadmat(stim_fname)["stim_labels"][0]

        print(f"Loaded {len(stim_labels)} stimulus labels")
        print(f"{sum(stim_labels == 3)} hand trials (out of {len(stim_labels)})")

        stc_epo_array = np.nanmean(
            stc_epo[stim_labels == 3], axis=0
        )  # average over hand trials

        assert isinstance(stc_epo_array, np.ndarray), "Input must be an array"
        return stc_epo_array

    def _load_complete_data(
        self,
        subjects: Union[Subject, Group],
    ):
        assert isinstance(subjects, Subject) or isinstance(
            subjects, Group
        ), "Input must be an instance of Subject or Group"

        if isinstance(subjects, Group):
            subjects_list = [subject for subject in subjects.subjects]
            print(f"Loading data for {len(subjects_list)} subjects...")
            print(f"Subjects: {[subject.subject_id for subject in subjects_list]}")
        elif isinstance(subjects, Subject):
            subjects_list = [subjects]

        epochs_data_arrays = []
        sem_epochs_per_sub = []
        stc_epo_arrays = []
        stc_eyes_open_arrays = []
        for subject in subjects_list:
            this_sub_id = subject.subject_id
            epochs, epochs, sem = self._load_epochs(this_sub_id)
            epochs_data_arrays.append(epochs)
            sem_epochs_per_sub.append(sem)

            stc_epo_array = self._load_stc_epochs(this_sub_id)
            stc_epo_arrays.append(stc_epo_array)

            stc_eyes_open = None
            stc_eyes_open_arrays.append(stc_eyes_open)

        # combine data across subjects
        stc_epo_array = np.nanmean(np.array(stc_epo_arrays), axis=0)
        if stc_epo_array.ndim != 3:
            stc_epo_array = np.expand_dims(stc_epo_array, axis=0)
        stc_eyes_open = (
            np.nanmean(np.array(stc_eyes_open_arrays), axis=0)
            if stc_eyes_open is not None
            else None
        )
        epochs_data_arrays = np.array(epochs_data_arrays)
        sem_epochs_per_sub = np.array(sem_epochs_per_sub)

        return (
            epochs,
            epochs_data_arrays,
            sem_epochs_per_sub,
            stc_epo_array,
            stc_eyes_open,
        )

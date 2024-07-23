import os
import numpy as np
import mne
from src.utils.config import Config
import src.configs.config as configs
from mne.datasets import fetch_fsaverage

config = Config.from_json(configs.CFGLog)

RESAMPLE_FREQ = config.parameters.sfreq
RANDOM_STATE = config.parameters.random_seed

mne.set_log_level("WARNING")

# Source the fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subject = "fsaverage"
trans = "fsaverage"
subjects_dir = os.path.dirname(fs_dir)
src = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")  # surface for dSPM
bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
model_fname = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem.fif")
snr = 1.0  # for non-averaged data


def apply_inverse(
    mne_object,
    inverse_operator,
    labels,
    sub_id,
    method,
    average_dipoles=True,
):
    """
    Apply inverse operator to MNE object and save STC files.

    Args:
        mne_object (object): The MNE object to apply inverse operator to.
        inverse_operator (object): The inverse operator to be used.
        labels (list): The labels to extract time courses from.
        sub_id (str): The subject ID.
        average_dipoles (bool, optional): Whether to average dipoles. Defaults to True.

    Returns:
        tuple: A tuple containing the label time courses and the subject ID if label time courses contain NaN values.
    """
    apply_inverse_kwargs = dict(
        lambda2=1.0 / snr**2,
        verbose=True,
    )
    stc_epochs = None
    if isinstance(mne_object, mne.io.fiff.raw.Raw) or isinstance(
        mne_object, mne.io.array.array.RawArray
    ):
        print("Applying inverse to Raw object")
        stc = mne.minimum_norm.apply_inverse_raw(
            mne_object, inverse_operator, method=method, **apply_inverse_kwargs
        )
    elif isinstance(mne_object, mne.Epochs) or isinstance(
        mne_object, mne.epochs.EpochsArray
    ):
        print("Applying inverse to Epochs object")
        stc = mne.minimum_norm.apply_inverse_epochs(
            mne_object, inverse_operator, method=method, **apply_inverse_kwargs
        )
    else:
        raise ValueError("Invalid mne_object type")

    # Extract labels and do mean flip
    src = inverse_operator["src"]
    mode = "mean_flip" if average_dipoles else None
    stc_epochs = mne.extract_label_time_course(stc, labels, src, mode=mode)

    return stc_epochs


def compute_fwd_and_inv(
    sub_id,
    snr,
    trans,
    src,
    bem,
    mne_object,
    noise_var,
    labels,
    method,
    average_dipoles=True,
):
    """
    Save the time course data for specified labels.

    Parameters:
        sub_id (str): The subject ID.
        snr (float): The signal-to-noise ratio.
        trans (str): The path to the transformation matrix file.
        src (str): The path to the source space file.
        bem (str): The path to the BEM model file.
        mne_object (MNE object): The MNE object containing the data.
        noise_var (MNE object): The noise covariance matrix.
        labels (list of str): The labels to save the time course for.
        save_path (str): The path to save the time course data.
        average_dipoles (bool, optional): Whether to average dipoles (default: True).

    Returns:
        None
    """
    stc_epochs = None  # Initialize variables
    fwd = mne.make_forward_solution(
        mne_object.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        n_jobs=-1,
        verbose=True,
    )

    inverse_operator = mne.minimum_norm.make_inverse_operator(
        mne_object.info, fwd, noise_var, verbose=True
    )

    stc_epochs = apply_inverse(
        mne_object,
        inverse_operator,
        labels,
        sub_id,
        method=method,
        average_dipoles=True,
    )

    return stc_epochs


def source_localize(
    eyes_open,
    sub_id,
    epochs=None,
    roi_names=config.parameters.roi_names,
    average_dipoles=True,
    method="MNE",
    return_eyes_open=False,
):
    """
    Compute the source localization for a subject for eyes closed, eyes open, and z-scored epochs.
    Args:
        sub_id (str): The ID of the subject.
        processed_data_path (str): The path to the data.
        zscored_epochs_save_path (str): The path to save the Z-scored epochs.
        EC_resting_save_path (str): The path to save the EC resting state data.
        EO_resting_save_path (str): The path to save the EO resting state data.
        roi_names (list): The names of the ROIs.
        times_tup (tuple): A tuple containing the time window information.
        return_zepochs (bool, optional): Whether to return the Z-scored epochs. Defaults to True.
        return_EC_resting (bool, optional): Whether to return the EC resting state data. Defaults to True.
        return_eyes_open: (bool, optional): Whether to return the EO resting state data. Defaults to True.
        average_dipoles (bool, optional): Whether to average the dipoles. Defaults to True.

    Returns:
        None
    """

    # Convert ROI names to labels
    labels = [
        mne.read_labels_from_annot(subject, regexp=roi, subjects_dir=subjects_dir)[0]
        for roi in roi_names
    ]

    # Compute noise & data covariance
    noise_cov = mne.compute_raw_covariance(eyes_open, verbose=True)
    noise_cov = mne.cov.regularize(noise_cov, eyes_open.info, eeg=0.1, verbose=True)

    # Create the noise covariance
    data = np.diag(np.diag(noise_cov.data))
    names = eyes_open.info["ch_names"]
    bads = eyes_open.info["bads"]
    projs = eyes_open.info["projs"]
    nfree = data.shape[0]

    # keep only good channels
    good_names = [name for name in names if name not in bads]

    # Extract the diagonal elements
    noise_var = mne.Covariance(
        data=data,
        names=good_names,
        bads=bads,
        projs=projs,
        nfree=nfree,
        verbose=True,
    )

    # Preallocate
    stc_object = None
    # If desired and eyes open resting data not yet processed, process it
    if return_eyes_open:
        stc_object = compute_fwd_and_inv(
            sub_id,
            snr,
            trans,
            src,
            bem,
            eyes_open,
            noise_var,
            labels,
            method,
            average_dipoles=True,
        )
    else:
        print("Source localizing epochs...")
        stc_object = compute_fwd_and_inv(
            sub_id,
            snr,
            trans,
            src,
            bem,
            epochs,
            noise_var,
            labels,
            method,
            average_dipoles=True,
        )

    return np.asarray(stc_object)
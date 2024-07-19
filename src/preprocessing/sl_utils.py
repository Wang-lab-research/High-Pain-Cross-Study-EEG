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


def apply_inverse_and_save(
    mne_object,
    inverse_operator,
    labels,
    save_path,
    save_fname,
    sub_id,
    condition,
    method,
    average_dipoles=True,
    save_stc_mat=False,
):
    """
    Apply inverse operator to MNE object and save STC files.

    Args:
        mne_object (object): The MNE object to apply inverse operator to.
        inverse_operator (object): The inverse operator to be used.
        labels (list): The labels to extract time courses from.
        save_path (str): The path to save the STC files.
        sub_id (str): The subject ID.
        condition (str): The condition.
        average_dipoles (bool, optional): Whether to average dipoles. Defaults to True.

    Returns:
        tuple: A tuple containing the label time courses and the subject ID if label time courses contain NaN values.
    """
    apply_inverse_and_save_kwargs = dict(
        lambda2=1.0 / snr**2,
        verbose=True,
    )
    stc_epochs = None
    if isinstance(mne_object, mne.io.fiff.raw.Raw):
        print("Applying inverse to Raw object")
        stc = mne.minimum_norm.apply_inverse_raw(
            mne_object, inverse_operator, method=method, **apply_inverse_and_save_kwargs
        )
    elif isinstance(mne_object, mne.epochs.EpochsArray):
        print("Applying inverse to Epochs object")
        stc = mne.minimum_norm.apply_inverse_epochs(
            mne_object, inverse_operator, method=method, **apply_inverse_and_save_kwargs
        )
    else:
        raise ValueError("Invalid mne_object type")

    # Extract labels and do mean flip
    src = inverse_operator["src"]
    mode = "mean_flip" if average_dipoles else None
    stc_epochs = mne.extract_label_time_course(stc, labels, src, mode=mode)

    return stc_epochs


def save(stc_epochs, as_mat=False):
    # Save as pickle
    if not as_mat:
        with open(os.path.join(save_path, save_fname), "wb") as file:
            pickle.dump(stc_epochs, file)
        print(f"Saved {save_fname} to {save_path}.")

    # Save Z-scored Epochs STC only. MAT file for analysis in MATLAB
    elif save_stc_mat and isinstance(stc_epochs, list):
        # Reshape for convention (optional)
        nepochs = len(stc_epochs)
        stc_epochs = np.concatenate(stc_epochs)
        stc_epochs = np.reshape(stc_epochs, (nepochs, len(labels), len(mne_object.times)))
        print("*stc_epochs shape = ", stc_epochs.shape)

        for i in range(len(labels)):
            print(f"Saving stc.mat for {sub_id} in region: {labels[i].name}")
            stc_epochs_i = stc_epochs[:, i, :]
            print("*stc_epochs_i shape = ", stc_epochs_i.shape)

            # Save STC Zepochs per region
            matfiledata = {"data": stc_epochs_i}
            save_fname = f"{labels[i].name}_{condition}.mat"
            # hdf5storage.write(
            #     matfiledata,
            #     filename=os.path.join(sub_save_path, save_fname),
            #     matlab_compatible=True,
            # )
            sub_save_path = os.path.join(save_path, sub_id)
            savemat(os.path.join(sub_save_path, save_fname), matfiledata)


def compute_fwd_and_inv(
    sub_id,
    condition,
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
        condition (str): The condition of the data.
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
    # Check if files already saved. Check already in place for regular save to pkl
    sub_done = False
    if save_stc_mat:
        sub_save_path = os.path.join(save_path, sub_id)
        if not os.path.exists(sub_save_path):
            os.makedirs(sub_save_path)
        if len(os.listdir(sub_save_path)) >= len(labels):
            sub_done = True

    stc_epochs, None  # Initialize variables
    if not sub_done:
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

        stc_epochs, sub_id_if_nan = apply_inverse_and_save(
            mne_object,
            inverse_operator,
            labels,
            sub_id,
            condition,
            method=method,
            average_dipoles=True,
        )

    return stc_epochs, 
    


def to_source(
    sub_id,
    processed_data_path,
    zscored_epochs_save_path,
    EC_resting_save_path,
    EO_resting_save_path,
    roi_names,
    times_tup,
    method,
    return_zepochs=True,
    return_EC_resting=False,
    return_EO_resting=False,
    average_dipoles=True,
    save_stc_mat=False,
    save_inv=True,
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
        return_EO_resting (bool, optional): Whether to return the EO resting state data. Defaults to True.
        average_dipoles (bool, optional): Whether to average the dipoles. Defaults to True.

    Returns:
        None
    """

    # Convert ROI names to labels
    labels = [
        mne.read_labels_from_annot(subject, regexp=roi, subjects_dir=subjects_dir)[0]
        for roi in roi_names
    ]

    # Extract time window information from tuple arguments
    tmin, tmax, bmax = times_tup

    # Compute noise & data covariance
    eo_segment = load_raw(processed_data_path, sub_id, condition="eyes_open")
    noise_cov = mne.compute_raw_covariance(eo_segment, verbose=True)
    # Regularize the covariance matrices
    noise_cov = mne.cov.regularize(noise_cov, eo_segment.info, eeg=0.1, verbose=True)

    # Create the noise covariance
    data = np.diag(np.diag(noise_cov.data))
    names = eo_segment.info["ch_names"]
    bads = eo_segment.info["bads"]
    projs = eo_segment.info["projs"]
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

    #################################################################################################

    # TODO: Control regions only?
    control_regions = False

    # If processing resting, check directories for count
    raw = load_raw(processed_data_path, sub_id, condition="preprocessed")
    if return_EO_resting:
        raw_eo = load_raw(processed_data_path, sub_id, condition="eyes_open")
        EO_save_fname = (
            f"{sub_id}_eyes_open.pkl"
            if not control_regions
            else f"{sub_id}_eyes_open_control.pkl"
        )
    if return_EC_resting:
        raw_ec = load_raw(processed_data_path, sub_id, condition="eyes_closed")
        EC_save_fname = (
            f"{sub_id}_eyes_closed.pkl"
            if not control_regions
            else f"{sub_id}_eyes_closed_control.pkl"
        )
    # If processing epochs, check directory for count
    if return_zepochs:
        zepochs_save_fname = (
            f"{sub_id}_epochs.pkl"
            if not control_regions
            else f"{sub_id}_epochs_control.pkl"
        )

    # Preallocate
    stc_epochs_EO, stc_epochs_EC, stc_epochs_Epochs = None, None, None
    sub_id_if_nan = None
    # If desired and eyes open resting data not yet processed, process it
    if return_EO_resting and not os.path.exists(
        f"{EO_resting_save_path}/{EO_save_fname}"
    ):
        stc_epochs_EO, sub_id_if_nan = compute_fwd_and_inv(
            sub_id,
            "EO",
            snr,
            trans,
            src,
            bem,
            raw_eo,
            noise_var,
            labels,
            EO_resting_save_path,
            EO_save_fname,
            method=method,
            average_dipoles=True,
            save_stc_mat=save_stc_mat,
            save_inv=save_inv,
        )

    # If desired and eyes closed resting data not yet processed, process it
    if return_EC_resting and not os.path.exists(
        f"{EC_resting_save_path}/{EC_save_fname}"
    ):
        stc_epochs_EC, sub_id_if_nan = compute_fwd_and_inv(
            sub_id,
            "EC",
            snr,
            trans,
            src,
            bem,
            raw_ec,
            noise_var,
            labels,
            EC_resting_save_path,
            EC_save_fname,
            method=method,
            average_dipoles=True,
            save_stc_mat=save_stc_mat,
            save_inv=save_inv,
        )

    # If desired and epochs not yet processed, Z-score and source localize
    if return_zepochs:
        if not save_stc_mat and not os.path.exists(
            f"{zscored_epochs_save_path}/{zepochs_save_fname}"
        ):
            print("Z-scoring epochs...")
            zepochs = zscore_epochs(sub_id, processed_data_path, tmin, raw)
            # print shape of zepochs
            print(zepochs.get_data().shape)

            print("Source localizing epochs...")
            stc_epochs_Epochs, sub_id_if_nan = compute_fwd_and_inv(
                sub_id,
                "epochs",
                snr,
                trans,
                src,
                bem,
                zepochs,
                noise_cov,
                labels,
                zscored_epochs_save_path,
                zepochs_save_fname,
                method=method,
                average_dipoles=True,
                save_stc_mat=save_stc_mat,
                save_inv=save_inv,
            )
        if save_stc_mat:  # for save mat overwrite existing folder
            print(zscored_epochs_save_path, zepochs_save_fname)
            print(os.path.exists(f"{zscored_epochs_save_path}/{zepochs_save_fname}"))

            print("Z-scoring epochs...")
            zepochs = zscore_epochs(sub_id, processed_data_path, tmin, raw)
            # print shape of zepochs
            print(zepochs.get_data().shape)

            print("Source localizing epochs...")
            stc_epochs_Epochs, sub_id_if_nan = compute_fwd_and_inv(
                sub_id,
                "epochs",
                snr,
                trans,
                src,
                bem,
                zepochs,
                noise_cov,
                labels,
                zscored_epochs_save_path,
                zepochs_save_fname,
                method=method,
                average_dipoles=True,
                save_stc_mat=save_stc_mat,
                save_inv=save_inv,
            )

    return (stc_epochs_Epochs, stc_epochs_EO, stc_epochs_EC), sub_id_if_nan

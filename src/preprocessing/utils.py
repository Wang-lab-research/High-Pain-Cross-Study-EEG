import os
import glob


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
    data_files += glob.glob(subject_folder + "/*.EDF")
    if len(data_files) != 1:
        raise ValueError(
            f"Expected one EDF file in {subject_folder}, found {len(data_files)}"
        )
    return data_files[0]

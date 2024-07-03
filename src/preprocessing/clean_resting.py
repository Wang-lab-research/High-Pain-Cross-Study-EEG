import os
import mne
import pandas as pd
from pathlib import Path
from src.utils.config import Config
from src.configs.config import CFGLog

# Settings
times_tup,time_win_path = utils.get_time_window(5)

# Data paths
data_dir = Path('../../Data/')
data_path = data_dir / 'EEG DATA/'
processed_data_path = data_dir / 'Processed Data/'
csv_path = data_dir / 'Eyes Timestamps/'
epo_path = processed_data_path 
save_path_resting = processed_data_path

save_paths=[save_path_resting,]
[os.makedirs(path, exist_ok=True) for path in save_paths];


sub_ids = sub_ids_CP + sub_ids_HC + sub_ids_WSP + sub_ids_LP

# keep unique only
sub_ids = list(set(sub_ids))

# %%
print(f"Chronics: {len([el for el in sub_ids if el.startswith('0')])}")
print(f"Controls: {len([el for el in sub_ids if el.startswith('C')])}")
print(f"Total: {len(sub_ids)}")

# %%
# Include noise? if so EO is 3 minutes long. If not, EO is 5 minutes long
include_noise = False

if include_noise:
    save_path = processed_data_path
else:
    save_path = processed_data_path / '5min'
    os.makedirs(save_path, exist_ok=True)

# %% [markdown]
# ### Run just the 5 min resting crop

# %%
for sub_id in sub_ids:
    # Load raw
    raw = mne.io.read_raw_fif(epo_path / f'{sub_id}_preprocessed-raw.fif', preload=True)

    # Preprocess continuous to eyes open, noise calibration, and eyes closed

    # Check if file already exists
    if os.path.isfile(save_path / f'{sub_id}_eyes_open-raw.fif'):
        print(f"{sub_id} already processed")
        continue
    
    _, _, _, = preprocess.get_cropped_resting_EEGs(sub_id, 
                                                    raw,
                                                    csv_path, 
                                                    save_path, 
                                                    include_noise=include_noise)

# %%
for sub_id in sub_ids:
    # Preprocess continuous to eyes open, noise calibration, and eyes closed
    raw, eyes_closed, noise_segment, eyes_open = preprocess.to_raw(data_path,
                                                                   sub_id,
                                                                   save_path=save_path,
                                                                   csv_path=csv_path,
                                                                   include_noise=include_noise)
    # raw = mne.io.read_raw_fif(epo_path / f'{sub_id}_preprocessed-raw.fif', preload=True)
    
    stim_epochs = mne.read_epochs(epo_path / f'{sub_id}_preprocessed-epo.fif')
    # stim_epochs, epo_times, stim_labels, pain_ratings = preprocess.to_epo(raw, sub_id, data_path, save_path=processed_data_path) 
    
    # check epochs for duration in seconds
    dur = stim_epochs.times[-1] - stim_epochs.times[0]
    print(f"{sub_id} duration: {dur}")
    if dur < 5.0:
        print(f"{sub_id} too short")
        break

# %%




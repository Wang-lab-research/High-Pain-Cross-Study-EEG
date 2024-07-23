import os

AUTHOR = "George Kenefati"

PARENT_PATH = "../../../../"

CFGLog = {
    "data": {
        "chronic_low_back_pain": {
            "subject_ids": {
                "CP": [
                    "018",
                    "022",
                    "024",
                    "031",
                    "032",
                    "034",
                    "036",
                    "039",
                    "040",
                    "045",
                    "046",
                    "052",
                    "020",
                    "021",
                    "023",
                    "029",
                    "037",
                    "041",
                    "042",
                    "044",
                    "048",
                    "049",
                    "050",
                    "056",
                ],
                "HC": [
                    "C10",
                    "C11",
                    "C12",
                    "C13",
                    "C14",
                    "C15",
                    "C16",
                    "C17",
                    "C18",
                    "C19",
                    "C2.",
                    "C24",
                    "C25",
                    "C26",
                    "C27",
                    "C3.",
                    "C6.",
                    "C7.",
                    "C9.",
                ],
            },
            "path": os.path.abspath(
                os.path.join(
                    PARENT_PATH, f"{AUTHOR}/Chronic Low Back Pain Study/Data/Raw/"
                )
            ),
            "processed_path": os.path.abspath(
                os.path.join(
                    PARENT_PATH,
                    f"{AUTHOR}/Chronic Low Back Pain Study/Data/Processed Data/",
                )
            ),
            "epochs_path": os.path.abspath(
                os.path.join(
                    PARENT_PATH, f"{AUTHOR}/Chronic Low Back Pain Study/Data/Epochs/"
                )
            ),
        },
        "chronic_pancreatitis": {
            "subject_ids": {
                "CP": [
                    "002",
                    "003",
                    "006",
                    "007",
                    "008",
                    "009",
                    "010",
                    # "011", conflated pain of hiccups with pinpricks
                    "012",
                    "013",
                    "014",
                ],
            },
            "path": os.path.abspath(
                os.path.join(PARENT_PATH, f"{AUTHOR}/Pancreatitis Pain Study/Data/Raw/")
            ),
        },
        "lupus": {
            "subject_ids": {
                "CP": [
                    "5186",
                    "6310",
                    "5295",
                ],
                "NP": [
                    "5873",
                    "6100",
                    "6106",
                    "5648",
                    "5675",
                    "5845",
                    "5713",
                ],
            },
            "path": os.path.abspath(
                os.path.join(PARENT_PATH, f"{AUTHOR}/Lupus EEG Biomarker/Data/Raw/")
            ),
        },
    },
    "parameters": {
        "sfreq": 600,
        "random_seed": 42,
        "tmin": -2.0,
        "tmax": 1.0,
        "roi_names": [  # Left
            "rostralanteriorcingulate-lh",  # Left Rostral ACC
            "caudalanteriorcingulate-lh",  # Left Caudal ACC
            "postcentral-lh",  # Left S1,
            "insula-lh",
            "superiorfrontal-lh",  # Left Insula, Left DL-PFC,
            "medialorbitofrontal-lh",  # Left Medial-OFC
            # Right
            "rostralanteriorcingulate-rh",  # Right Rostral ACC
            "caudalanteriorcingulate-rh",  # Right Caudal ACC
            "postcentral-rh",  # , Right S1
            "insula-rh",
            "superiorfrontal-rh",  # Right Insula, Right DL-PFC
            "medialorbitofrontal-rh",  # Right Medial-OFC",
        ],
        "32_channel_montage_file_path": os.path.abspath(
            os.path.join(
                PARENT_PATH,
                f"{AUTHOR}/Code/eeg_toolkit/montages/Hydro_Neo_Net_32_xyz_cms_No_Fp1.sfp",
            )
        ),
        "64_channel_montage_file_path": os.path.abspath(
            os.path.join(
                PARENT_PATH,
                f"{AUTHOR}/Code/eeg_toolkit/montages/Hydro_Neo_Net_64_xyz_cms.sfp",
            )
        ),
        "ch_names": [
            "Fp1",
            "Fpz",
            "Fp2",
            "AF3",
            "AF4",
            "F11",
            "F7",
            "F5",
            "F3",
            "F1",
            "Fz",
            "F2",
            "F4",
            "F6",
            "F8",
            "F12",
            "FT11",
            "FC5",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "FC6",
            "FT12",
            "T7",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "T8",
            "TP7",
            "CP5",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "CP6",
            "TP8",
            "M1",
            "M2",
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
            "PO7",
            "PO3",
            "POz",
            "PO4",
            "PO8",
            "O1",
            "Oz",
            "O2",
            "Cb1",
            "Cb2",
        ],
        "roi_acronyms": [
            "L_rACC",
            "R_dACC",
            "L_S1",
            "L_Ins",
            "L_dlPFC",
            "L_mOFC",
            "R_rACC",
            "R_dACC",
            "R_S1",
            "R_Ins",
            "R_dlPFC",
            "R_mOFC",
        ],
        "freq_bands": {
            "theta": [4.0, 8.0],
            "alpha": [8.0, 13.0],
            "beta": [13.0, 30.0],
            "low-gamma": [30.0, 58.5],
            "high-gamma": [61.5, 100.0],
        },
        "event_id_all": {
            "eyes_closed": {
                "eyes closed": 1,
                "Trigger#1": 1,
                "EYES CLOSED": 1,
            },
            "eyes_open": {
                "eyes open": 2,
                "eyes opened": 2,
                "Trigger#2": 2,
                "EYES OPEN": 2,
                "eyes openned": 2,
            },
            "high_hand": {
                "pinprick hand": 3,
                "hand pinprick": 3,
                "Yes Pain Hand": 3,
                "Trigger#3": 3,
                "HAND PINPRICK": 3,
                "hand 32 gauge pinprick": 3,
                "Yes Hand Pain": 3,
                "Hand YES Pain prick": 3,
            },
            "med_hand": {
                "Med Pain Hand": 4,
                "Med Hand Pain": 4,
                "Hand Medium Pain prick": 4,
            },
            "low_hand": {
                "No Pain Hand": 5,
                "hand plastic": 5,
                "plastic hand": 5,
                "Trigger#4": 5,
                "HAND PLASTIC": 5,
                "hand plastic filament": 5,
                "No Hand Pain": 5,
                "Hand NO Pain": 5,
            },
            "high_back": {
                "pinprick back": 6,
                "back pinprick": 6,
                "Yes Pain Back": 6,
                "BACK  PINPRICK": 6,
                "BACK PINPRICK": 6,
                "Trigger#5": 6,
                "back 32 gauge pinprick": 6,
                "Yes Back Pain": 6,
                "Back YES Pain prick": 6,
            },
            "med_back": {
                "Med Pain Back": 7,
                "Med Back Pain": 7,
                "Back Medium Pain prick": 7,
            },
            "low_back": {
                "plastic back": 8,
                "back plastic": 8,
                "No Pain Back": 8,
                "BACK PLASTIC": 8,
                "Trigger#6": 8,
                "back plastic filament": 8,
                "No Back Pain": 8,
                "Back No Pain": 8,
            },
            "stop": {
                "stop": 9,
                "Stop": 9,
                "STOP": 9,
            },
            "pinprick_markers": {
                "1000001": 10,
                "100160": 10,
                "100480": 10,
                "1000000": 10,
                "1000010": 11,
                "100048": 11,
                "1100001": 12,
                "100320": 12,
                "1100010": 13,
            },
        },
        "event_id": {
            "hand_high_stim": 3,
            "hand_med_stim": 4,
            "hand_low_stim": 5,
            "back_high_stim": 6,
            "back_med_stim": 7,
            "back_low_stim": 8,
            "pinprick10": 10,
            "pinprick11": 11,
            "pinprick12": 12,
            "pinprick13": 13,
        },
    },
    "output": {
        "parent_save_path": os.path.abspath("../../data/preprocessed/"),
        "parent_stc_save_path": {
            "eyes_open": os.path.abspath(
                "../../data/preprocessed/source_time_courses/eyes_open"
            ),
            "epochs": os.path.abspath(
                "../../data/preprocessed/source_time_courses/epochs/{PERISTIM_TIME_WIN}_sec_time_window"
            ),
        },
    },
}

import os

AUTHOR = "George Kenefati"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_PATH = "../../../../"

PERISTIM_TIME_WIN = 5

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
            "path": os.path.join(
                PARENT_PATH, f"{AUTHOR}/Chronic Low Back Pain Study/Data/Raw/"
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
                    "011",
                    "012",
                    "013",
                ],
            },
            "path": os.path.join(
                PARENT_PATH, f"{AUTHOR}/Pancreatitis Pain Study/Data/Raw/"
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
            "path": os.path.join(
                PARENT_PATH, f"{AUTHOR}/Lupus EEG Biomarker/Data/Raw/"
            ),
        },
    },
    "parameters": {
        "sfreq": 600,
        "random_seed": 42,
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
    },
    "output": {
        "parent_save_path": "../../data/preprocessed/",
        "parent_stc_save_path": {
            "eyes_open": "../../data/preprocessed/source_time_courses/eyes_open",
            "epochs": "../../data/preprocessed/source_time_courses/epochs/{PERISTIM_TIME_WIN}_sec_time_window",
        },
    },
}

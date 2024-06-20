CFGLog = {
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
            "WSP": [
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
                    ],
            "LP": [
                    "020",
                    "021",
                    "023",
                    "029",
                    "044",
                    "037",
                    "041",
                    "042",
                    "048",
                    "049",
                    "050",
                    "056",
                    ],
        },

        "path": "./data/chronic_low_back_pain.csv",
    },    
    
    "chronic_pancreatitis": {
        "subject_ids": {
            "CP": [],
            "HC": [],
            "WSP": [],
        },

        "path": "./data/chronic_pancreatitis.csv",
    },
    
    "lupus": {
        "subject_ids": {
            "CP": [],
            "HC": [],
            "WSP": [],
        },

        "path": "./data/lupus.csv",
    },
    
    "train": {
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": 2022,
        "C": 0.01,
        "penalty": "l2",
    },
    
    "data_info": {
        "sfreq": 600,
        "roi_names": [# Left
             'rostralanteriorcingulate-lh', # Left Rostral ACC
             'caudalanteriorcingulate-lh', # Left Caudal ACC
             'postcentral-lh', # Left S1,
             'insula-lh', 'superiorfrontal-lh', # Left Insula, Left DL-PFC,
             'medialorbitofrontal-lh', # Left Medial-OFC
             # Right
             'rostralanteriorcingulate-rh', # Right Rostral ACC
             'caudalanteriorcingulate-rh', # Right Caudal ACC
             'postcentral-rh', # , Right S1
             'insula-rh', 'superiorfrontal-rh', # Right Insula, Right DL-PFC
             'medialorbitofrontal-rh', # Right Medial-OFC",
            ],
        "roi_acronyms": ["L_rACC", "R_dACC", 
                         "L_S1", "L_Ins", 
                         "L_dlPFC", "L_mOFC",
                         "R_rACC", "R_dACC",
                         "R_S1", "R_Ins",
                         "R_dlPFC", "R_mOFC"],
        
        "freq_bands": {
            "theta": [4.0, 8.0],
            "alpha": [8.0, 13.0],
            "beta": [13.0, 30.0],
            "low-gamma": [30.0, 58.5],
            "high-gamma": [61.5, 100.0],
        },
    },
    
    
        
    "output": {
        "output_path": "./data/exported_models/",
        "model_name": "20230217_152148_LogReg.pickle",
    },
}
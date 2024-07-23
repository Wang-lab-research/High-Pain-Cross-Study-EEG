%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% State space model  (SSM) and Kalman filter (KF) for detecting acute pain
% signals based on Human EEG data
close all; clc; format short; format compact; tic; 
clear data_windows S_range S_entire y_range y_entire trainY;
addpath((genpath('./matlab packages/chronux_2_12/')));
addpath((genpath('./matlab packages/lds/')));
addpath((genpath('./matlab packages/mne-matlab-master/matlab/')));
addpath((genpath('./matlab packages/LFP_code/'))); 
addpath((genpath('./utils/'))); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parameters

% LFP omits S1, EEG uses sensor space, STC uses source ROIs
modality = "stc";

if modality=="lfp"
    roi_names = {'left','right'}; 
elseif modality=="eeg"
    roi_names = {'F3','AF3','C3','CP1'};
elseif modality=="stc"
    roi_names = {
        "rostralanteriorcingulate-lh",  % Left Rostral ACC
        "caudalanteriorcingulate-lh",   % Left Caudal ACC
        "postcentral-lh",               % Left S1
        "insula-lh",                    
        "superiorfrontal-lh",           % Left Insula, Left DL-PFC
        "medialorbitofrontal-lh"        % Left Medial-OFC
        "rostralanteriorcingulate-rh",  % Right Rostral ACC
        "caudalanteriorcingulate-rh",   % Right Caudal ACC
        "postcentral-rh",               % Right S1
        "insula-rh",
        "superiorfrontal-rh",           % Right Insula, Right DL-PFC
        "medialorbitofrontal-rh"        % Right Medial-OFC
    };
end

fband_struct = struct(...
    'theta', [4, 8], ...    
    'alpha', [8, 13], ...    
    'beta', [13, 30], ...        
    'low_gamma', [30, 58.5], ... 
    'high_gamma', [61.5, 100] ...  
);

Fs=600; % sampling rate

%% Thresholds for binarizing pain
pain_threshold = 4;
gap = 1;
high_pain_threshold = 8; % threshold for train trials

%% Training parameters
Tbin                =       2; % seconds before and after events
train_plot_flag     =       0; % plot or not
log_transform_flag  =       1; % instead of mean, helps to compare high and low freqs
fft_flag            =       0; % fft instead of mtspecgram, maybe defunct?
folds               =       5; % number of train trials to attempt

%% Load data

if modality=="stc"
    proc_dir = "High-Pain-Cross-Study-EEG/data/preprocessed/"; % processed data
    sub_num = "039";
end

abs_proc_dir = fullfile(pwd, proc_dir);
addpath(genpath(abs_proc_dir)); 
stc_epochs_dir = fullfile(abs_proc_dir, sub_num+"_stc_epochs/");
events_path = fullfile(abs_proc_dir, sub_num+"_events.mat");
pain_ratings_path = fullfile(abs_proc_dir, sub_num+"_pain_ratings.mat");
stimulus_labels_path = fullfile(abs_proc_dir, sub_num+"_stimulus_labels.mat");

stc_epochs = struct();
mat_files = dir(fullfile(stc_epochs_dir, '*.mat'));

% Load each MAT file into the stc_epochs struct
for i = 1:length(mat_files)
    file_name = mat_files(i).name;
    region_name = erase(file_name, '.mat'); % Extract the region name without the file extension
    region_name = strrep(region_name, '-', '_'); % Replace hyphens with underscores
    file_path = fullfile(stc_epochs_dir, file_name);
    data = load(file_path);
    stc_epochs.(region_name) = data.data;
end

% Display the contents of stc_epochs
events = load(events_path);
ratings = load(pain_ratings_path).data;
stimulus_labels = load(stimulus_labels_path);

%% Ensure epochs, ratings, events have the same length
if length(stc_epochs) ~= length(ratings) || length(stc_epochs) ~= length(events) ...
        || length(stc_epochs) ~= length(stimulus_labels)
    error('STC Epochs, ratings, and events have different lengths');
end

%% Convert pain ratings to binary based on a threshold and a gap
ratings_bin = double((ratings > pain_threshold) | (ratings < pain_threshold - gap));

%% Omit channels with low SnR
% 
% if modality_flag==2 % EEG
%     [raw_data,selected_goods] = drop_bad_chs(loaded_data.raw_struct,...
%         loaded_data.raw_data,roi_names);
% else
%     selected_goods = 1:length(roi_names);
% end

%% Select hand, back, or both
% [epo_times,stim_labels,pain_ratings,stim_and_ratings,~,LS_ids,HS_ids] = keep_trials(loaded_data.epo_times,...
%     loaded_data.stim_labels,loaded_data.pain_ratings,modality_flag);
% stim_ids.LS_ids = LS_ids; % save these for selecting trials for train/test 
% stim_ids.HS_ids = HS_ids;

%% Select trials for training

openvar('stim_and_ratings');commandwindow % bring focus back to CW
% train_ids  = input("Please enter the indices of five trials with high pain ratings for training:\n>> "); 
% test only on non-train ids
test_ids = 1:length(stim_and_ratings);
test_ids = setdiff(test_ids,train_ids);

%% Set up training data

close all;
[trainY,fbands,T,tt_entire] = train_dataloader(Fs,selected_goods,epo_times,...
    raw_data,train_ids,fft_flag,log_transform_flag,Tbin,modality_flag,train_plot_flag);

%% Off-line EM algorithm for LDS Estimation

LS_flag             =       1; % 1: include, 0: exclude LS as stimulus events with HS 
ccf_thresh_flag     =       2; % 2: 2*std(ccf_value), otherwise custom value
save_best_flag      =       0; % 0: save all trials, 1: save only the "best"

[SSMtrain_ACC,SSMtrain_S1,best_el,best_ff,fband_perms,opt_params,F1_scores_train,CCF_value_train] = EM_LDS_estimator(Fs, ...
    fbands,fband_names,epo_times,raw_data,trainY,train_ids, ...
    stim_ids,T,tt_entire,modality_flag,LS_flag,ccf_thresh_flag,save_best_flag);
% pause

%% Online KF -- CONTINUOUS TRACE

close all;
KF_plot_flag        =       1; % plot KF detections or not
median_flag         =       1; % use median or mean on state-space before z_score
ccf_thresh_flag     =       2; % 2: 2*std(ccf_value), otherwise custom value
if modality_flag~=1 % if not LFP
    test_model_flag = input('1: Q0 = SSMtrain_ACC.Q; 2: Q0 = SSMtrain_S1.Q\n>> ');
end
test_model_flag=1;
[~,CCF_value_test] = KF_estimator(SSMtrain_ACC,SSMtrain_S1, ...
    best_el,best_ff,opt_params,Fs,fbands,fband_perms,fband_names,epo_times, ...
    raw_data,train_ids,trainY,T,tt_entire,stim_ids,LS_flag,median_flag, ...
    KF_plot_flag,ccf_thresh_flag,modality_flag,test_model_flag,save_best_flag);
% openvar('F1_scores_test')

%% Online KF -- LOOP with ONE training trial

close all;
[F1_scores_test] = KF_estimator_LOOP(SSMtrain_ACC,SSMtrain_S1,opt_params,Fs,fbands, ...
    fband_perms,fband_names,epo_times,raw_data,train_ids,trainY,T,tt_entire, ...
    stim_ids,LS_flag,median_flag,KF_plot_flag,ccf_thresh_flag,modality_flag);

%% Online KF -- LOOP with ITERATIVE training and testing

[F1_scores_test] = KF_estimator_LOOP_ITER(SSMtrain_ACC,SSMtrain_S1,opt_params,Fs,fbands, ...
    fband_perms,fband_names,epo_times,raw_data,train_ids,trainY,T,tt_entire, ...
    stim_ids,LS_flag,median_flag,KF_plot_flag,ccf_thresh_flag,modality_flag);

%% save the raw_data data around the TP and FP events
% 
% t1 = Tbin; % save X s before and X s after TP and FP events
% t2 = Tbin; % ^
% for n = 1:Nchs % doesn't include S1-L/R!
%     stim_rep_TP = 1;
%     for i = 2:length(TP_ind) % first is for training SSM
%         onset_TP = detection(TP_ind(i)); % TP time (not index)
%         ind_TP = round(onset_TP * Fs); % TP time as raw_data signal index 
%         range_TP = [ind_TP - t1*Fs : ind_TP + t2*Fs - 1];
%         raw_data_TP = raw_data(n).raw_data(range_TP);
%         Data_Stim_TP(n,:,stim_rep_TP) = raw_data_TP;
%         stim_rep_TP = stim_rep_TP+1;
%     end
% 
%     stim_rep_FP = 1;
%     for i = 2:length(FP_ind)
%         onset_FP = detection(FP_ind(i)); % FP time (not index)
%         ind_FP = round(onset_FP * Fs); % FP time as raw_data signal index 
%         range_FP = [ind_FP - t1*Fs : ind_FP + t2*Fs - 1];
%         raw_data_FP = raw_data(n).raw_data(range_FP);
%         Data_Stim_FP(n,:,stim_rep_FP) = raw_data_FP;
%         stim_rep_FP = stim_rep_FP+1;
%     end
% end

toc; % return run time

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% State space model  (SSM) and Kalman filter (KF) for detecting acute pain
% signals based on Human EEG data
close all; clc; format short; format compact; tic; 
clear
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
params = get_params(modality);
Fs = params.Fs;

%% Load data
if modality=="stc"
    proc_dir = "../../data/preprocessed/"; % processed data
    subject_id = "039";
end

[stc_epochs, events, ratings, stimulus_labels] = load_stc_mat(proc_dir, subject_id);

%% Ensure epochs, ratings, events have the same length
if size(stc_epochs,1) ~= size(ratings,1) || size(stc_epochs,1) ~= size(events,1) ...
        || size(stc_epochs,1) ~= size(stimulus_labels,1)
    error('STC Epochs, ratings, and events have different lengths');
end

%% Convert pain ratings to binary based on a threshold and a gap
ratings_bin = double((ratings > params.pain_threshold) | (ratings < params.pain_threshold - params.gap));

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
high_pain_trials = find(ratings > params.high_pain_threshold);

if isempty(high_pain_trials)
    error('No trials with ratings greater than the high pain threshold were found.');
end

%% Initialize variables to store results
precisions = [];
recalls = [];
f1s = [];
confusion_matrices = [];
aucs = [];
    
%% Loop through high pain trials to get best for training
for i = 1:params.folds
    random_idx = randi(length(high_pain_trials));
    train_trial_id = high_pain_trials(random_idx);
    disp(['Random trial ID with high pain rating: ', num2str(train_trial_id)]);
    
    train_trial = squeeze(stc_epochs(train_trial_id,:,:));

    test_trial_ids = 1:size(stc_epochs,1);
    test_trial_ids(train_trial_id) = [];
    test_trials = stc_epochs(test_trial_ids, :, :);
    test_labels = ratings(test_trial_ids);

    %% Set up training data
    close all;
    crop_flag = 1; % whether to crop trials shorter
    tmin = -1; tmax = 1;
    [trainY,fbands,T,tt_entire] = extract_features(Fs,params.roi_names,events,...
        train_trial,train_trial_id,params.fft_flag,params.log_transform_flag,tmin,tmax,crop_flag,params.train_plot_flag);

    %% TRAINING: Off-line EM algorithm for LDS Estimation
    
    ccf_thresh_flag     =       2; % 2: 2*std(ccf_value), otherwise custom value
    save_best_flag      =       0; % 0: save all trials, 1: save only the "best"
    
    [SSMtrain_ACC,SSMtrain_S1,best_el,best_ff,fband_perms,opt_params,F1_scores_train,CCF_value_train] = EM_LDS_estimator(Fs, ...
        fbands,fband_names,epo_times,raw_data,trainY,train_ids, ...
        stim_ids,T,tt_entire,modality_flag,LS_flag,ccf_thresh_flag,save_best_flag);
    % pause
    
    %% TESTING: Online KF -- CONTINUOUS TRACE
    
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
end

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

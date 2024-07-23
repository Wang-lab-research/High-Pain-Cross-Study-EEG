function [F1_scores_test,CCF_value_test] = KF_estimator(SSMtrain_ACC,SSMtrain_S1,best_el, ...
    best_ff,opt_params,Fs,fbands,fband_perms,fband_names,epo_times,raw_data, ...
    train_ids,trainY,T,tt_entire,stim_ids,LS_flag,median_flag,KF_plot_flag, ...
    ccf_thresh_flag,modality_flag,test_model_flag,save_best_flag)
%KF_ESTIMATOR detect pain based on SSM

%% Display
clc; disp('Performing online KF estimation...');

%% Main
rng(0);

if modality_flag~=1 % EEG or STC, include S1
    % import opt_params
    opt_a=opt_params(1);opt_b=opt_params(2);opt_p=opt_params(3);opt_ccf=opt_params(4);

    % read model
    z0 = 0;
    if test_model_flag==1
        Q0 = SSMtrain_ACC.Q; 
    elseif test_model_flag==2

        Q0 = SSMtrain_S1.Q;
    end
    
    %% Test on best model or on all training trials
    if save_best_flag==1
    if best_ff <= length(fbands)
        fprintf('\nFband index: %s\n',fband_names(best_ff))
        [state_ACC, ~] = KF_filter(SSMtrain_ACC, trainY{1,3}(:,best_ff)', z0, Q0); % 3 is whole trace for ACC
        [state_S1, ~] = KF_filter(SSMtrain_S1, trainY{1,4}(:,best_ff)', z0, Q0); % 4 is whole trace for S1
        % best train trial
        [state_ACC_train, ~] = KF_filter(SSMtrain_ACC, trainY{best_el,1}(:,best_ff)', z0, Q0);
        [state_S1_train, ~] = KF_filter(SSMtrain_S1, trainY{best_el,2}(:,best_ff)', z0, Q0);
    
    % for the rest, test each combination of two fbands
    elseif best_ff > length(fbands)
        ff_adj=best_ff-5;
        fprintf('\nFband indices: %s + %s\n',[fband_names(fband_perms(ff_adj,1)),fband_names(fband_perms(ff_adj,2))] );
        trainY_freq_perm_tmp_ACC = squeeze([trainY{1,3}(:,fband_perms(ff_adj,1)),trainY{1,3}(:,fband_perms(ff_adj,2))]);
        trainY_freq_perm_tmp_S1 = squeeze([trainY{1,4}(:,fband_perms(ff_adj,1)),trainY{1,4}(:,fband_perms(ff_adj,2))]);
        [state_ACC, ~] = KF_filter(SSMtrain_ACC, trainY_freq_perm_tmp_ACC', z0, Q0); % 3 is whole trace for ACC
        [state_S1, ~] = KF_filter(SSMtrain_S1, trainY_freq_perm_tmp_S1', z0, Q0); % 4 is whole trace for S1
    
        % best train trial
        trainY_freq_perm_tmp_ACC_train = squeeze([trainY{best_el,1}(:,fband_perms(ff_adj,1)),trainY{best_el,1}(:,fband_perms(ff_adj,2))]);
        trainY_freq_perm_tmp_S1_train = squeeze([trainY{best_el,2}(:,fband_perms(ff_adj,1)),trainY{best_el,2}(:,fband_perms(ff_adj,2))]);
        [state_ACC_train, ~] = KF_filter(SSMtrain_ACC, trainY_freq_perm_tmp_ACC_train', z0, Q0); % 3 is whole trace for ACC
        [state_S1_train, ~] = KF_filter(SSMtrain_S1, trainY_freq_perm_tmp_S1_train', z0, Q0); % 4 is whole trace for S1
    end

    % Z-score the state output based on the pre-stim baseline
    train_baseline = 1:(T-1)/2;

    if median_flag
        mean_Z_ACC = median(state_ACC_train(train_baseline));
        std_Z_ACC = std(state_ACC_train(train_baseline));
        meanK_Z_S1 = median(state_S1_train(train_baseline));
        stdK_Z_S1 = std(state_S1_train(train_baseline));
    else
        mean_Z_ACC = mean(state_ACC_train(train_baseline));
        std_Z_ACC = std(state_ACC_train(train_baseline));
        meanK_Z_S1 = mean(state_S1_train(train_baseline));
        stdK_Z_S1 = std(state_S1_train(train_baseline));
    end
    
    tem_Z_ACC = (state_ACC' - mean_Z_ACC) * 1.5/ std_Z_ACC;
    temK_Z_S1 = (state_S1' - meanK_Z_S1) * 1.5/ stdK_Z_S1;
    
    CCF_value = ccf_cal_opt(tem_Z_ACC,temK_Z_S1,opt_a,opt_b,opt_p);

    % Compute CCF
    if ccf_thresh_flag==2
        ccf_thresh_new = 2*std(CCF_value);
    else %
        ccf_thresh_new = ccf_thresh_flag; 
    end

    ccf_thresh_new = ccf_thresh_new + opt_ccf; % adjustable threshold
%     CCF_bin = abs(CCF_value) - ccf_thresh_new; % abs value for -ccf_thresh cross 
    CCF_bin = CCF_value - ccf_thresh_new; % abs value for -ccf_thresh cross 
          
    % Detection analysis
    CCF_bin(CCF_bin>=0) = 1;
    CCF_bin(CCF_bin<0) = -1; 
    CCF_diff = [0;diff(CCF_bin)];
   
    % Use only desired events (HS or HS and LS)
    if LS_flag==1
        stim_event_ids = [stim_ids.HS_ids;stim_ids.LS_ids];
    else
        stim_event_ids = stim_ids.HS_ids;
    end

    detections = tt_entire(CCF_diff>0); % cross-threshold times
    epo_event_times = epo_times/Fs; % samples rather than time   
    [TP_ind,FP_ind] = FP_check2(detections,epo_event_times); % T or F detection idxs
    
    % Report performance
    TP_times = detections(TP_ind); %length(TP_times)
    FP_times = detections(FP_ind);

    % metrics
    Precision = length(TP_times)/(length(TP_times)+length(FP_times)); % # of TP / # of Detections
    % correct rate
    CR = length(TP_times)/length(stim_event_ids); % # of TP / # of Stim events
    FP_overall = length(FP_times)/( length(raw_data)/Fs/60 ); % # of FP / # of total mins 
    % in entire recording.

    vdfvlnn
    wanglab00







        
    wanglab007
    
    % Metrics used to check best train trial
    fprintf('\nF1-Score:\t%3.3f\n',F1_score);
    fprintf('FP/min:\t\t%3.3f\n',FP_overall); 
    fprintf('\nCCF threshold: %f\n',ccf_thresh_new);
    fprintf('***********************************\n')

    F1_scores_test = F1_score;
    CCF_value_test = CCF_value;
%     TP_times_test = TP_times;
%     FP_times_test = FP_times;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if KF_plot_flag==1
        % PLOT DETECTION TIMES OVER CCF_VALUE
    
        % figure and position
        f = figure('Name','ONLINE Estimates w/ Detections','NumberTitle','off');close(f);
        f = figure('Name','ONLINE Estimates w/ Detections','NumberTitle','off');close(f);
        f.Position = [720 1100 2000 500];
        
        % SSMtrain ACC KF Zscore 
        subplot(311)
        sgtitle('KF Estimates w/ Detections');
        hold on; plot(tt_entire,tem_Z_ACC,'b-','LineWidth',2);
        hold on; yline(ccf_thresh_new,'k--','linewidth',1);
        hold on; yline(-ccf_thresh_new,'k--','linewidth',1);
        
        % event times 
        for ii = 1:length(stim_event_ids) % for each stim time 
            hold on; xline(stim_event_ids(ii)/Fs,'k-','LineWidth',2);
        end
    
        % include TP and FPs
        for tp = 1:length(TP_times)
            hold on; xline(TP_times(tp),'g-','LineWidth',3);
        end
        for fp = 1:length(FP_times)
            hold on; xline(FP_times(fp),'r-','LineWidth',1);
        end
        ylabel('KF ACC Z-score','FontSize',16);
        set(gca,'FontSize',16);
        ylim([-7 7]); xlim([0 2000]);
        
        % SSMtrain S1 KF Zscore 
        subplot(312)
        hold on; plot(tt_entire,temK_Z_S1,'b-','LineWidth',2);
        hold on; yline(ccf_thresh_new,'k--','linewidth',1);
        hold on; yline(-ccf_thresh_new,'k--','linewidth',1);
        
        % event times 
        for ii = 1:length(stim_event_ids) % for each stim time 
            hold on; xline(stim_event_ids(ii)/Fs,'k-','LineWidth',2);
        end
    
        % include TP and FPs
        for tp = 1:length(TP_times)
            hold on; xline(TP_times(tp),'g-','LineWidth',3);
        end
        for fp = 1:length(FP_times)
            hold on; xline(FP_times(fp),'r-','LineWidth',1);
        end
        ylabel('KF S1 Z-score','FontSize',16);
        set(gca,'FontSize',16);
        ylim([-7 7]); xlim([0 2000]);
    
        % KF CCF
        subplot(313);
        hold on; plot(tt_entire,CCF_value,'b-','LineWidth',2);
        hold on; yline(ccf_thresh_new,'k--','linewidth',1);
        hold on; yline(-ccf_thresh_new,'k--','linewidth',1);
    
        % event times 
        for ii = 1:length(stim_event_ids) % for each stim time 
            hold on; xline(stim_event_ids(ii)/Fs,'k-','LineWidth',2);
        end
    
        % include TP and FPs
        for tp = 1:length(TP_times)
            hold on; xline(TP_times(tp),'g-','LineWidth',3);
        end
        for fp = 1:length(FP_times)
            hold on; xline(FP_times(fp),'r-','LineWidth',1);
        end
        ylabel('KF CCF','FontSize',16);
        set(gca,'FontSize',16);
        ylim([-7 7]); xlim([0 2000]);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

elseif modality_flag==1 % LFP, omit S1
    % read model
    z0 = 0;
    Q0 = SSMtrain_ACC.Q; 

    if best_ff <= length(fbands)
        fprintf('\nFband index: %s\n',fband_names(best_ff))
        [state_ACC, ~] = KF_filter(SSMtrain_ACC, trainY{1,3}(:,best_ff)', z0, Q0); % 3 is whole trace for ACC
        % best train trial
        [state_ACC_train, ~] = KF_filter(SSMtrain_ACC, trainY{best_el,1}(:,best_ff)', z0, Q0);
    
    % for the rest, test each combination of two fbands
    elseif best_ff > length(fbands)
        ff_adj=best_ff-5;
        fprintf('\nFband indices: %s + %s\n',[fband_names(fband_perms(ff_adj,1)),fband_names(fband_perms(ff_adj,2))] );
        trainY_freq_perm_tmp_ACC = squeeze([trainY{1,3}(:,fband_perms(ff_adj,1)),trainY{1,3}(:,fband_perms(ff_adj,2))]);
        [state_ACC, ~] = KF_filter(SSMtrain_ACC, trainY_freq_perm_tmp_ACC', z0, Q0); % 3 is whole trace for ACC
    
        % best train trial
        trainY_freq_perm_tmp_ACC_train = squeeze([trainY{best_el,1}(:,fband_perms(ff_adj,1)),trainY{best_el,1}(:,fband_perms(ff_adj,2))]);
        [state_ACC_train, ~] = KF_filter(SSMtrain_ACC, trainY_freq_perm_tmp_ACC_train', z0, Q0); % 3 is whole trace for ACC
    end

    % Z-score the state output based on the pre-stim baseline
    train_baseline = 1:(T-1)/2;

    if median_flag
        mean_Z_ACC = median(state_ACC_train(train_baseline));
        std_Z_ACC = std(state_ACC_train(train_baseline));
    else
        mean_Z_ACC = mean(state_ACC_train(train_baseline));
        std_Z_ACC = std(state_ACC_train(train_baseline));
    end    

    tem_Z_ACC = (state_ACC' - mean_Z_ACC) * 1.5/ std_Z_ACC;
      
    CCF_value=tem_Z_ACC;

    % Compute CCF
    if ccf_thresh_flag==2
        ccf_thresh_new = 2*std(CCF_value);
    else %
        ccf_thresh_new = ccf_thresh_flag; 
    end
    %             CCF_bin = abs(CCF_value) - ccf_thresh_new; % abs value for -ccf_thresh cross 
    CCF_bin = CCF_value - ccf_thresh_new; % abs value for -ccf_thresh cross 
    
    
    % Detection analysis
    CCF_bin(CCF_bin>=0) = 1;
    CCF_bin(CCF_bin<0) = -1; 
    CCF_diff = [0;diff(CCF_bin)];
    
    % Use only desired events (HS or HS and LS)
    if LS_flag==1
        stim_event_ids = [stim_ids.HS_ids;stim_ids.LS_ids];
    else
        stim_event_ids = stim_ids.HS_ids;
    end

    detections = tt_entire(CCF_diff>0); % cross-threshold times
    epo_event_times = epo_times(stim_event_ids)/Fs; % samples rather than time   
    [TP_ind,FP_ind] = FP_check2(detections,epo_event_times); % T or F detection idxs
    
    % Report performance
    TP_times = detections(TP_ind);
    FP_times = detections(FP_ind);
    
    % metrics
    Precision = length(TP_times)/(length(TP_times)+length(FP_times)); % # of TP / # of Stims
    
    CR = length(TP_times)/length(stim_event_ids); % # of TP / # of Stim events
    F1_score = 2 * ( ( Precision*CR) / (Precision+CR) );

    FP_overall = length(FP_times)/( length(raw_data)/Fs/60 ); % FP/min
    % in entire recording.

    % Metrics used to check best train trial
    fprintf('\nF1-Score:\t%3.3f\n',F1_score);
    fprintf('FP/min:\t\t%3.3f\n',FP_overall); 
    fprintf('\nCCF threshold: %f\n',ccf_thresh_new);
    fprintf('***********************************\n')

    F1_scores_test = F1_score;
    CCF_value_test = CCF_value;
%     TP_times_test = TP_times;
%     FP_times_test = FP_times;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if KF_plot_flag==1
        % PLOT DETECTION TIMES OVER CCF_VALUE
    
        % figure and position
        fig = figure('Name','ONLINE Estimates w/ Detections','NumberTitle','off');close(fig);
        fig = figure('Name','ONLINE Estimates w/ Detections','NumberTitle','off');
        fig.Position = [720 1100 2000 500];
        
        % SSMtrain ACC KF Zscore 
        sgtitle('KF Estimates w/ Detections');
        hold on; plot(tt_entire,tem_Z_ACC,'b-','LineWidth',2);
        hold on; yline(ccf_thresh_new,'k--','linewidth',1);
        hold on; yline(-ccf_thresh_new,'k--','linewidth',1);
        
        % event times 
        for ii = 1:length(epo_event_times) % for each stim time 
            hold on; xline(epo_event_times(ii),'k-','LineWidth',2);
        end
    
        % include TP and FPs
        for tp = 1:length(TP_times)
            hold on; xline(TP_times(tp),'g-','LineWidth',3);
        end
        for fp = 1:length(FP_times)
            hold on; xline(FP_times(fp),'r-','LineWidth',1);
        end
        ylabel('KF ACC Z-score','FontSize',16);
        set(gca,'FontSize',16);
    %     ylim([-7 7]); xlim([0 2000]);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

end % modality_flag
end % func
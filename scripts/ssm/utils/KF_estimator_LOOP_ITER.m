function [KF_metrics] = KF_estimator(SSMtrain_ACC,SSMtrain_S1,opt_params,Fs,fbands,fband_perms,fband_names,epo_times,raw_data,train_ids,trainY,T,tt_entire,stim_ids,LS_flag,median_flag,KF_plot_flag,ccf_thresh_flag,modality_flag)
%KF_ESTIMATOR detect pain based on SSM

%% Display
clc; disp('Performing online KF estimation...');

%% Main
rng(0);

if modality_flag~=1 % EEG or STC, include S1
    % import opt_params
    opt_a=opt_params(1);opt_b=opt_params(2);opt_p=opt_params(3);opt_ccf=opt_params(4);

    for el=1:length(train_ids)
        k=train_ids(el);
        % time range
        range = 1:(T-1)/2;
        
        for ff = 1:(length(fbands)+length(fband_perms))
            z0 = 0;
            Q0 = SSMtrain_ACC.Q; 
        
            % test different frequency feature combinations iteratively
            % for the first five, test each frequency band on its own
            if ff <= length(fbands)
                fprintf('\nFband index: %s\n',fband_names(ff))
                [state_ACC, ~] = KF_filter(SSMtrain_ACC, trainY{1,3}(:,ff)', z0, Q0); % 3 is whole trace for ACC
                [state_S1, ~] = KF_filter(SSMtrain_S1, trainY{1,4}(:,ff)', z0, Q0); % 4 is whole trace for S1
            % for the rest, test each combination of two fbands
            elseif ff > length(fbands)
                ff_adj=ff-5;
                fprintf('\nFband indices: %s+%s\n',[fband_names(fband_perms(ff_adj,1)),fband_names(fband_perms(ff_adj,2))] );
                trainY_freq_perm_tmp_ACC = squeeze([trainY{1,3}(:,fband_perms(ff_adj,1)),trainY{1,3}(:,fband_perms(ff_adj,2))]);
                trainY_freq_perm_tmp_S1 = squeeze([trainY{1,4}(:,fband_perms(ff_adj,1)),trainY{1,4}(:,fband_perms(ff_adj,2))]);
                [state_ACC, ~] = KF_filter(SSMtrain_ACC, trainY_freq_perm_tmp_ACC', z0, Q0); % 3 is whole trace for ACC
                [state_S1, ~] = KF_filter(SSMtrain_S1, trainY_freq_perm_tmp_S1', z0, Q0); % 4 is whole trace for S1
            end
        
            % Z-score the state output based on the pre-stim baseline
            if median_flag
                mean_Z_ACC = median(state_ACC(range));
                std_Z_ACC = std(state_ACC(range));
                meanK_Z_S1 = median(state_S1(range));
                stdK_Z_S1 = std(state_S1(range));
            else
                mean_Z_ACC = mean(state_ACC(range));
                std_Z_ACC = std(state_ACC(range));
                meanK_Z_S1 = mean(state_S1(range));
                stdK_Z_S1 = std(state_S1(range));
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
%             CCF_bin = abs(CCF_value) - ccf_thresh_new; % abs value for -ccf_thresh cross 
            CCF_bin = CCF_value - ccf_thresh_new; % abs value for -ccf_thresh cross 
                  
            % Detection analysis
        
            CCF_bin(CCF_bin>=0) = 1;
            CCF_bin(CCF_bin<0) = -1; 
            CCF_diff = [0;diff(CCF_bin)];
            
            detections = tt_entire(CCF_diff>0); % cross-threshold times
            epo_event_times = epo_times/Fs; % samples rather than time   
            [TP_ind,FP_ind] = FP_check2(detections,epo_event_times); % T or F detection idxs
            
            % Report performance
            TP_times = detections(TP_ind); %length(TP_times)
            FP_times = detections(FP_ind);

            % metrics
            Precision = length(TP_times)/(length(TP_times)+length(FP_times)); % # of TP / # of Stims
            
            % Calculate recall only with desired events
            if LS_flag==1
                stim_event_ids = [stim_ids.HS_ids;stim_ids.LS_ids];
            else
                stim_event_ids = stim_ids.HS_ids;
            end

            Recall = length(TP_times)/length(stim_event_ids); % # of TP / # of Stim events
            F1_score = 2 * ( ( Precision*Recall) / (Precision+Recall) );

            FP_overall = length(FP_times)/( length(raw_data)/Fs/60 ); % # of FP / # of total mins 
            % in entire recording.
    
            % Metrics used to check best train trial
            fprintf('\nF1-Score:\t%3.3f\n',F1_score);
            fprintf('FP/min:\t\t%3.3f\n',FP_overall); 
            fprintf('***********************************')

            FP_overalls(el,ff) = FP_overall;
            F1_scores(el,ff) = F1_score;
            TP_times_all{el,ff} = TP_times;
            FP_times_all{el,ff} = FP_times;

        end
    end
    
    if KF_plot_flag==1
        f = figure(1999);close(f);f=figure(1999);
        f.Position = [1040 100 2400 1400];
        y_label = 'CCF Z-score';
        x_label = 'Time (s)';  
        
        sgtitle('ONLINE Decoder Detections around Stim Times');
        % xlabels = {'-100','0','100','200','300','400','500','600'};
        
        for i=1:length(epo_times)
            h = subplot(10,6,i); rngSz=5;% how much to view before and after
            stim_range_sec = floor(epo_times(i)-rngSz):floor(epo_times(i)+rngSz);
            % account for 50ms windows
            cnvrt = Fs/50; % conversion factor
            stim_range_win = floor((epo_times(i)-rngSz)*cnvrt):floor((epo_times(i)+rngSz)*cnvrt);
        
            hold on; plot(tt_entire(stim_range_win),CCF_value(stim_range_win),'b-','LineWidth',2);
            hold on; yline(ccf_thresh_new,'k--','linewidth',1);
            hold on; yline(-ccf_thresh_new,'k--','linewidth',1);
     
            % for plotting adaptive threshold (defunct)
    %         hold on; plot(tt_entire(stim_range_win),ccf_threshK_adapt(stim_range_win),'k--','linewidth',1);
    %         hold on; plot(tt_entire(stim_range_win),-ccf_threshK_adapt(stim_range_win),'k--','linewidth',1);
        
            for sT = 1:length(epo_times) % for each stim time 
                hold on; xline(epo_times(i),'k-','LineWidth',2);
            end
    
    %         % include TP and FPs
    %         for tp = 1:length(TP_times)
    %             tp_check = TP_times(tp);
    %             if tp_check > stim_range_sec(1) && tp_check < stim_range_sec(end)
    %                 hold on; xline(TP_times(tp),'g-','LineWidth',2);
    %             end
    %         end
    %         for fp = 1:length(FP_times)
    %             fp_check = FP_times(fp);
    %             if fp_check > stim_range_sec(1) && fp_check < stim_range_sec(end)
    %                 hold on; xline(FP_times(fp),'r-','LineWidth',2);
    %             end
    %         end
    %         
            ylabel(y_label);
    %         set(gca,'FontSize',16);
            xlim([stim_range_sec(1) stim_range_sec(end)]); ylim([-7 7]); 
            title(['stim ',num2str(i)]);
        end
    end
    
    %% re-plot KF output with detected TP and FP
    
    if KF_plot_flag==1
        f = figure('Name',[sprintf('ONLINE Estimates w/ Detections: Trial %d',k)],'NumberTitle','off');close(f);
        f = figure('Name',[sprintf('ONLINE Estimates w/ Detections: Trial %d',k)],'NumberTitle','off');
        f.Position = [720 1100 2000 500];
        subplot(311) % SSMtrain KF Zscore 
        sgtitle([sprintf('KF Estimates w/ Detections: Trial %d',k)]);
        hold on; plot(tt_entire,tem_Z_ACC,'b-','LineWidth',2);
        hold on; yline(3.85,'k--','linewidth',1);
        hold on; yline(-3.85,'k--','linewidth',1);
        for sT = 1:length(epo_times) % for each stim time 
            hold on; xline(epo_times(sT),'k-','LineWidth',2);
        end
        % include TP and FPs
        for tp = 1:length(TP_times)
            hold on; xline(TP_times(tp),'g-','LineWidth',3);
        end
        for fp = 1:length(FP_times)
            hold on; xline(FP_times(fp),'r-','LineWidth',1);
        end
        ylabel('KF 1 Z-score','FontSize',16);
        set(gca,'FontSize',16);
        ylim([-7 7]); xlim([0 2000]);
        
        subplot(312) % SSMtrain2 KF Zscore 
        hold on; plot(tt_entire,temK_Z_S1,'b-','LineWidth',2);
        hold on; yline(3.85,'k--','linewidth',1);
        hold on; yline(-3.85,'k--','linewidth',1);
        for sT = 1:length(epo_times) % for each stim time 
            hold on; xline(epo_times(sT),'k-','LineWidth',2);
        end
        % include TP and FPs
        for tp = 1:length(TP_times)
            hold on; xline(TP_times(tp),'g-','LineWidth',3);
        end
        for fp = 1:length(FP_times)
            hold on; xline(FP_times(fp),'r-','LineWidth',1);
        end
        ylabel('KF 2 Z-score','FontSize',16);
        set(gca,'FontSize',16);
        ylim([-7 7]); xlim([0 2000]);
        
        subplot(313); % CCF of KF 1 and 2
        hold on; plot(tt_entire,CCF_value,'b-','LineWidth',2);
        hold on; yline(ccf_thresh_new,'k--','linewidth',1);
        hold on; yline(-ccf_thresh_new,'k--','linewidth',1);
        for sT = 1:length(epo_times) % for each stim time 
            hold on; xline(epo_times(sT),'k-','LineWidth',2);
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
        ylim([-7 7]); %xlim([0 2000]);
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

elseif modality_flag==1 % LFP, omit S1
    for el=1:length(train_ids)
        k=train_ids(el);
        % time range
        range = 1:(T-1)/2;
        
        for ff = 1:(length(fbands)+length(fband_perms))
            fprintf('\n***********************************')
            fprintf('\n------------- TRIAL %d ------------\n',k)

            z0 = 0;
            Q0 = SSMtrain_ACC.Q; 
        
            % test different frequency feature combinations iteratively
            % for the first five, test each frequency band on its own
            if ff <= length(fbands)
                fprintf('\nFBAND: %s\n',fband_names(ff))
                [state_ACC, ~] = KF_filter(SSMtrain_ACC, trainY{1,3}(:,ff)', z0, Q0); % 3 is whole trace for ACC
            % for the rest, test each combination of two fbands
            elseif ff > length(fbands)
                ff_adj=ff-5;
                fprintf('\nFBAND(S): %s + %s\n',[fband_names(fband_perms(ff_adj,1)),fband_names(fband_perms(ff_adj,2))] );
                trainY_freq_perm_tmp_ACC = squeeze([trainY{1,3}(:,fband_perms(ff_adj,1)),trainY{1,3}(:,fband_perms(ff_adj,2))]);
                [state_ACC, ~] = KF_filter(SSMtrain_ACC, trainY_freq_perm_tmp_ACC', z0, Q0); % 3 is whole trace for ACC
            end
        
            % Z-score the state output
            if median_flag
                mean_Z_ACC = median(state_ACC(range));
                std_Z_ACC = std(state_ACC(range));
            else
                mean_Z_ACC = mean(state_ACC(range));
                std_Z_ACC = std(state_ACC(range));
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
            
            detections = tt_entire(CCF_diff>0); % cross-threshold times
            epo_event_times = epo_times/Fs; % samples rather than time   
            [TP_ind,FP_ind] = FP_check2(detections,epo_event_times); % T or F detection idxs
            
            % Report performance
            TP_times = detections(TP_ind);
            FP_times = detections(FP_ind);
            
            % metrics
            Precision = length(TP_times)/(length(TP_times)+length(FP_times)); % # of TP / # of Stims
            
            % Calculate recall only with desired events
            if LS_flag==1
                stim_event_ids = [stim_ids.HS_ids;stim_ids.LS_ids];
            else
                stim_event_ids = stim_ids.HS_ids;
            end

            Recall = length(TP_times)/length(stim_event_ids); % # of TP / # of Stim events
            F1_score = 2 * ( ( Precision*Recall) / (Precision+Recall) );

            FP_overall = length(FP_times)/( length(raw_data)/Fs/60 ); % # of FP / # of total mins 
            % in entire recording.
    
            % Metrics used to check best train trial
            fprintf('\nF1-Score:\t%3.3f\n',F1_score);
            fprintf('FP/min:\t\t%3.3f\n',FP_overall); 
            fprintf('***********************************')

            FP_overalls(el,ff) = FP_overall;
            F1_scores(el,ff) = F1_score;
            TP_times_all{el,ff} = TP_times;
            FP_times_all{el,ff} = FP_times;

        end
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if KF_plot_flag==1
        f = figure(1998);close(f);f=figure(1998);
        f.Position = [720 100 2000 1000];
        y_label = 'CCF Z-score';
        x_label = 'Time (s)';  
        
        sgtitle('Decoder Detections around Stim Times OFFLINE');
%             xlabels = {'-100','0','100','200','300','400','500','600'};

        k=train_ids(el);
%                 for i=1:length(train_ids)
%                     h = subplot(2,3,i); % how much to view before and after
%                     stim_range_sec = floor(epo_times(i)-Tbin*Fs):floor(epo_times(i)+Tbin*Fs);
%                     
%                     % use fraction of entire duration to convert from epo_times
%                     % to tt_entire
%                     fraction_tmp = double(epo_times(i))/double(length(raw_data));
%                     tt_start = int64(fraction_tmp*length(tt_entire));
%                     stim_range_win = tt_start-T/2:tt_start+(T-1)/2;
%                 
%                     hold on; plot(tt_entire(stim_range_win),CCF_value_ACC,'b-','LineWidth',2);
%                     hold on; yline(ccf_thresh_ACC_new,'k--','linewidth',1);
%                     hold on; yline(-ccf_thresh_ACC_new,'k--','linewidth',1);
%             
%                     for sT = 1:length(epo_times) % for each stim time 
%                         hold on; xline(epo_times(i),'k-','LineWidth',2);
%                     end
%                 
%                     % include TP and FPs
%                     for tp = 1:length(TP_epo_times)
%                         tp_check = TP_times(tp);
%                         if tp_check > stim_range_sec(1) && tp_check < stim_range_sec(end)
%                             hold on; xline(TP_epo_times(tp),'g-','LineWidth',2);
%                         end
%                     end
%                     for fp = 1:length(FP_epo_times)
%                         fp_check = FP_epo_times(fp);
%                         if fp_check > stim_range_sec(1) && fp_check < stim_range_sec(end)
%                             hold on; xline(FP_epo_times(fp),'r-','LineWidth',2);
%                         end
%                     end
%                     
%                     ylabel(y_label);
%                     set(gca,'FontSize',13);
%                     ylim([-7 7]); xlim([stim_range_sec(1) stim_range_sec(end)]);
%                     title(['stim ',num2str(i)]);
%                 end
        
        % detected TP and FP
        f = figure('Name',[sprintf('Training Detections: Trial %d',k)],'NumberTitle','off');close(f);
        f = figure('Name',[sprintf('Training Detections: Trial %d',k)],'NumberTitle','off');
        f.Position = [720 1100 2000 150];
        hold on; plot(tt_entire,CCF_value_ACC,'b-','LineWidth',2);
        hold on; yline(ccf_thresh_ACC_new,'k--','linewidth',1);
        hold on; yline(-ccf_thresh_ACC_new,'k--','linewidth',1);
        for sT = 1:length(epo_times) % for each stim time 
            hold on; xline(epo_times(sT),'k-','LineWidth',2);
        end
        % include TP and FPs
        for tp = 1:length(TP_times)
            hold on; xline(TP_times(tp),'g-','LineWidth',3);
        end
        for fp = 1:length(FP_times)
            hold on; xline(FP_times(fp),'r-','LineWidth',1);
        end
        ylabel('OFFLINE CCF','FontSize',13);
        set(gca,'FontSize',13);
        ylim([-7 7]); %xlim([0 2000]);
            end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        f = figure(1999);close(f);f=figure(1999);
        f.Position = [1040 100 2400 1400];
        y_label = 'CCF Z-score';
        x_label = 'Time (s)';  
        
        sgtitle('ONLINE Decoder Detections around Stim Times');
%         xlabels = {'-100','0','100','200','300','400','500','600'};
        
        for i=1:length(epo_times)
            h = subplot(10,6,i); rngSz=5;% how much to view before and after
            stim_range_sec = floor(epo_times(i)-rngSz):floor(epo_times(i)+rngSz);
            % account for 50ms windows
            cnvrt = Fs/50; % conversion factor
            stim_range_win = floor((epo_times(i)-rngSz)*cnvrt):floor((epo_times(i)+rngSz)*cnvrt);
        
            hold on; plot(tt_entire(stim_range_win),CCF_value(stim_range_win),'b-','LineWidth',2);
            hold on; yline(ccf_thresh_new,'k--','linewidth',1);
            hold on; yline(-ccf_thresh_new,'k--','linewidth',1);
        
            for sT = 1:length(epo_times) % for each stim time 
                hold on; xline(epo_times(i),'k-','LineWidth',2);
            end
    
            % include TP and FPs
            for tp = 1:length(TP_times)
                tp_check = TP_times(tp);
                if tp_check > stim_range_sec(1) && tp_check < stim_range_sec(end)
                    hold on; xline(TP_times(tp),'g-','LineWidth',2);
                end
            end
            for fp = 1:length(FP_times)
                fp_check = FP_times(fp);
                if fp_check > stim_range_sec(1) && fp_check < stim_range_sec(end)
                    hold on; xline(FP_times(fp),'r-','LineWidth',2);
                end
            end
            
            ylabel(y_label);
    %         set(gca,'FontSize',16);
            xlim([stim_range_sec(1) stim_range_sec(end)]); ylim([-7 7]); 
            title(['stim ',num2str(i)]);
        end

    %% re-plot KF output with detected TP and FP
    
    if KF_plot_flag==1
        f = figure('Name',[sprintf('ONLINE Estimates w/ Detections: Trial %d',k)],'NumberTitle','off');close(f);
        f = figure('Name',[sprintf('ONLINE Estimates w/ Detections: Trial %d',k)],'NumberTitle','off');
        f.Position = [720 1100 2000 500];
        subplot(311) % SSMtrain KF Zscore 
        sgtitle([sprintf('KF Estimates w/ Detections: Trial %d',k)]);
        hold on; plot(tt_entire,tem_Z_ACC,'b-','LineWidth',2);
        hold on; yline(3.85,'k--','linewidth',1);
        hold on; yline(-3.85,'k--','linewidth',1);
        for sT = 1:length(epo_times) % for each stim time 
            hold on; xline(epo_times(sT),'k-','LineWidth',2);
        end
        % include TP and FPs
        for tp = 1:length(TP_times)
            hold on; xline(TP_times(tp),'g-','LineWidth',3);
        end
        for fp = 1:length(FP_times)
            hold on; xline(FP_times(fp),'r-','LineWidth',1);
        end
        ylabel('KF 1 Z-score','FontSize',16);
        set(gca,'FontSize',16);
        ylim([-7 7]); xlim([0 2000]);
    end
 
end
end % func
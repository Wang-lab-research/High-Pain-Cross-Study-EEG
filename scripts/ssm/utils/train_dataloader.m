function [trainY,fbands,T,tt_entire] = train_dataloader(Fs,selected_goods,epo_times,raw_data,train_ids,fft_flag,log_transform_flag,Tbin,modality_flag,train_plot_flag)
%TRAIN_DATALOADER read data directly from recordings

%% Display
clc; disp('Constructing train dataloader...');

%% Main
rng(0); % set seed

% No. of chs and trials
Nchs = length(selected_goods);

if modality_flag~=1 % EEG or STC, include S1
    % Preallocate continuous data matrix
    ACC_raw = zeros(Nchs/2,length(raw_data(1,:))); % 'F3','AF3',
    S1_raw = zeros(Nchs/2,length(raw_data(1,:))); % 'C3','CP1'
    
    for n=1:Nchs/2
        ACC_raw(n,:) = raw_data(selected_goods(n),:);
        S1_raw(n,:) = raw_data(selected_goods(n+Nchs/2),:);
    end
    
    % data params
    % Tbin = pre and post onset window in seconds
    
    % preallocate
    trainY = cell(length(train_ids),Nchs*2); % initiate Y matrix (# train trials x # chs)
    for el=1:size(trainY,1)
        k = train_ids(el);
        onset = epo_times(k);
    
        range = floor(onset-Tbin*Fs)+1 : floor(onset+Tbin*Fs); % [-Tbin, Tbin] s
    
        % calculate spectral power from raw_data signals
        for ch=1:Nchs/2
            % ACC
            data_windows(ch,:) = ACC_raw(ch,range); 
            % S1
            data_windows(ch+Nchs/2,:) = S1_raw(ch,range);
    
            if fft_flag == 1
                nfft = 2^nextpow2(movingwin(1)*Fs);
                % Tbin range
                % ACC
                [A11,f11,~] = spectrogram(data_windows(ch,:),movingwin(1)*Fs,(movingwin(1)-movingwin(2))*Fs,nfft,Fs);
                idx = intersect(find(f11>params.fpass(1)),find(f11<params.fpass(2)));
                f_fft = f11(idx)';
                S_temp = abs(A11(idx,:)').^2/nfft/(Fs/2);
                S_range(ch,:,:) = reshape(S_temp,[1,size(S_temp,1),size(S_temp,2)]);
    
                % S1
                [S11,f11,~] = spectrogram(data_windows(ch+Nchs/2,:),movingwin(1)*Fs,(movingwin(1)-movingwin(2))*Fs,nfft,Fs);
                idx = intersect(find(f11>params.fpass(1)),find(f11<params.fpass(2)));
                f_fft = f11(idx)';
                S_temp = abs(S11(idx,:)').^2/nfft/(Fs/2);
                S_range(ch+Nchs/2,:,:) = reshape(S_temp,[1,size(S_temp,1),size(S_temp,2)]);
    
                % entire recording range
                % ACC
                [AA11,f11,~] = spectrogram(ACC_raw(ch,:),movingwin(1)*Fs,(movingwin(1)-movingwin(2))*Fs,nfft,Fs);
                idx = intersect(find(f11>params.fpass(1)),find(f11<params.fpass(2)));
                f_fft = f11(idx)';
                SS_temp = abs(AA11(idx,:)').^2/nfft/(Fs/2);
                S_entire(ch,:,:) = reshape(SS_temp,[1,size(SS_temp,1),size(SS_temp,2)]);
    
                % S1
                [SS11,f11,~] = spectrogram(S1_raw(ch,:),movingwin(1)*Fs,(movingwin(1)-movingwin(2))*Fs,nfft,Fs);
                idx = intersect(find(f11>params.fpass(1)),find(f11<params.fpass(2)));
                f_fft = f11(idx)';
                SS_temp = abs(SS11(idx,:)').^2/nfft/(Fs/2);
                S_entire(ch+Nchs/2,:,:) = reshape(SS_temp,[1,size(SS_temp,1),size(SS_temp,2)]);
    
            else
                % mtspecgramc params
                movingwin = [0.250 0.050]; % [window size, step size] (seconds)
                params.Fs = Fs;
                params.tapers = [3 5];
                params.fpass=[4 100];
                params.pad = 0;
                params.trialave = 0;
                params.err = 0;
    
                % spectrogram
                [S_range(ch,:,:),~,~]=mtspecgramc(data_windows(ch,:),movingwin,params);
        
                [S_entire(ch,:,:),~,~]=mtspecgramc(ACC_raw(ch,:),movingwin,params);

                [S_range(ch+(Nchs/2),:,:),tt_range,~]=mtspecgramc(data_windows(ch+Nchs/2,:),movingwin,params);
        
                [S_entire(ch+(Nchs/2),:,:),tt_entire,f]=mtspecgramc(S1_raw(ch,:),movingwin,params);
            end
    
        end
    
        % set some more params
        edges = tt_range - Tbin; % times from mtspecgramc - Tbin = 'edges'
        T = length(tt_range);
        band_theta = find(f>4 & f<8); 
        band_alpha = find(f>8 & f<13); 
        band_beta = find(f>13 & f<30); 
        band_gammaL = find(f>30 & f<50);
        band_gammaH = find(f>70 & f<100); 
    
        fbands = {band_theta, band_alpha, band_beta, band_gammaL, band_gammaH};
    
        % log scale used for comparing low f to high f bands, like theta
        if log_transform_flag == 1
            para = 1; % make positive
            for i=1:Nchs % for each channel 
                for j=1:length(fbands) % for each frequency band
                    y_range(i,j,:) = (log10(sum(S_range(i,:, fbands{j})+para,3)));
                    % ensure y_entire is being created ONCE
                    if isempty(trainY{1,i+2})
                        y_entire(i,j,:) = (log10(sum(S_entire(i,:, fbands{j})+para,3)));
                    end
                end
            end
        else % average power over each frequency band:
            for i=1:Nchs % for each channel 
                for j=1:length(fbands) % for each frequency band
                    y_range(i,j,:) = mean((S_range(i,:, fbands{j})),3);
                    % ensure y_entire is being created ONCE
                    if isempty(trainY{1,i+2})
                        y_entire(i,j,:) = mean((S_entire(i,:, fbands{j})),3);
                    end
                end
            end
        end
    
        % demean the spectral power data
        for i=1:Nchs % for each channel 
            for j=1:length(fbands) % for each frequency band
                y_range(i,j,:) = y_range(i,j,:) - mean( y_range(i,j,:) );
                % ensure y_entire is being created ONCE
                if isempty(trainY{1,i+2})
                    y_entire(i,j,:) = y_entire(i,j,:) - mean( y_entire(i,j,:) );
                end
            end
        end
        
        % squeeze data together
        ch_ct = [1 3]; % for channel indexing
        for i=1:Nchs/2 
            % initiate with first frequency band, for each channel array
            Y_first_ch_range_tmp = [];
            Y_second_ch_range_tmp = [];

            % ensure y_entire is being created ONCE
            if isempty(trainY{1,i+2})
                Y_first_ch_entire_tmp = [];
                Y_second_ch_entire_tmp = [];
            end

            % stack array first in order of frequencies then by channel array
            for j=1:length(fbands)
                Y_first_ch_range_tmp = [Y_first_ch_range_tmp, y_range(ch_ct(i),j,:)];
                Y_second_ch_range_tmp = [Y_second_ch_range_tmp, y_range(ch_ct(i)+1,j,:)];
                
                % ensure y_entire is being created ONCE
                if isempty(trainY{1,i+2})
                    Y_first_ch_entire_tmp = [Y_first_ch_entire_tmp, y_entire(ch_ct(i),j,:)];
                    Y_second_ch_entire_tmp = [Y_second_ch_entire_tmp, y_entire(ch_ct(i)+1,j,:)];
                end
            end
    
            % order the first channel before the second (convention based on
            % original code)
            Y_range_tmp = [Y_first_ch_range_tmp, Y_second_ch_range_tmp];
            
            % ensure y_entire is being created ONCE
            if isempty(trainY{1,i+2})
                Y_entire_tmp = [Y_first_ch_entire_tmp, Y_second_ch_entire_tmp];
            end

            % squeeze
            trainY{el,i} = squeeze(Y_range_tmp)';
            
            % ensure y_entire is being created ONCE
            if isempty(trainY{1,i+2})
                trainY{1,i+2} = squeeze(Y_entire_tmp)';
            end    
        end
    
        if train_plot_flag
            % plot spectrogram
            fig = figure('Name',[sprintf('Spectrogram for Trial %d',k)],'NumberTitle','off');
            fig.Position = [720 1440-300*el-1 2000 250];
    
            subplot(211); % spectrogram of S1 channel 1
            plot_matrix(squeeze(S_range(3,:,:)),edges,f); xlabel('');
            hold on; plot(edges, squeeze(y_range(3,5,:)),'w-','LineWidth',2);
            hold on; xline(0,'k-','LineWidth',2);
            hold on; xline(1, 'r-','LineWidth',2);
            ylabel('S1 CH1','fontsize',16);
            xlim([-Tbin Tbin]), set(gca, 'fontsize',16); colorbar('off');
        
            subplot(212) % spectrogram of ACC channel 1
            plot_matrix(squeeze(S_range(1,:,:)),edges,f); xlabel('');
            hold on; plot(edges, squeeze(y_range(1,5,:)),'w-','LineWidth',2);
            hold on; xline(0,'k-','LineWidth',2);
            hold on; xline(1, 'r-','LineWidth',2);
            ylabel('ACC CH1','fontsize',16);
            xlim([-Tbin Tbin]), set(gca, 'fontsize',16); colorbar('off');
            xlabel('Time (sec)',fontsize=16);
        end
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif modality_flag==1 % LFP, omit S1
    % Preallocate continuous data matrix
    ACC_raw = zeros(Nchs,length(raw_data(1,:))); % 'F3','AF3',
    
    for n=1:Nchs
        ACC_raw(n,:) = raw_data(selected_goods(n),:);
    end
    
    % data params
    % Tbin = pre and post onset window in seconds
    
    % preallocate
    trainY = cell(length(train_ids),Nchs*2); % initiate Y matrix (# train trials x # chs)
    for el=1:length(train_ids)
        k = train_ids(el);
        onset = epo_times(k);
    
        range = floor(onset-Tbin*Fs)+1 : floor(onset+Tbin*Fs); % [-Tbin, Tbin] s
    
        % calculate spectral power from raw_data signals
        for ch=1:Nchs
            % ACC
            data_windows(ch,:) = ACC_raw(ch,range); 
    
            if fft_flag == 1
                nfft = 2^nextpow2(movingwin(1)*Fs);
                % Tbin range
                % ACC
                [A11,f11,~] = spectrogram(data_windows(ch,:),movingwin(1)*Fs,(movingwin(1)-movingwin(2))*Fs,nfft,Fs);
                idx = intersect(find(f11>params.fpass(1)),find(f11<params.fpass(2)));
                f_fft = f11(idx)';
                S_temp = abs(A11(idx,:)').^2/nfft/(Fs/2);
                S_range(ch,:,:) = reshape(S_temp,[1,size(S_temp,1),size(S_temp,2)]);
    
                % entire recording range
                % ACC
                [AA11,f11,~] = spectrogram(ACC_raw(ch,:),movingwin(1)*Fs,(movingwin(1)-movingwin(2))*Fs,nfft,Fs);
                idx = intersect(find(f11>params.fpass(1)),find(f11<params.fpass(2)));
                f_fft = f11(idx)';
                SS_temp = abs(AA11(idx,:)').^2/nfft/(Fs/2);
                S_entire(ch,:,:) = reshape(SS_temp,[1,size(SS_temp,1),size(SS_temp,2)]);
    
            else
                % mtspecgramc params
                movingwin = [0.250 0.050]; % [window size, step size] (seconds)
                params.Fs = Fs;
                params.tapers = [3 5];
                params.fpass=[4 100];
                params.pad = 0;
                params.trialave = 0;
                params.err = 0;
    
                % spectrogram
                [S_range(ch,:,:),tt_range,~]=mtspecgramc(data_windows(ch,:),movingwin,params);
                [S_entire(ch,:,:),tt_entire,f]=mtspecgramc(ACC_raw(ch,:),movingwin,params);   
            end
    
        end
    
        % set some more params
        edges = tt_range - Tbin; % times from mtspecgramc - Tbin = 'edges'
        T = length(tt_range);
        band_theta = find(f>4 & f<8); 
        band_alpha = find(f>8 & f<13); 
        band_beta = find(f>13 & f<30); 
        band_gammaL = find(f>30 & f<50);
        band_gammaH = find(f>70 & f<100); 
    
        fbands = {band_theta, band_alpha, band_beta, band_gammaL, band_gammaH};
    
        % log scale used for comparing low f to high f bands, like theta
        if log_transform_flag == 1
            para = 1; % make positive
            for i=1:Nchs % for each channel 
                for j=1:length(fbands) % for each frequency band
                    y_range(i,j,:) = (log10(sum(S_range(i,:, fbands{j})+para,3)));
                    % ensure y_entire is being created ONCE
                    if isempty(trainY{1,i+2})
                        y_entire(i,j,:) = (log10(sum(S_entire(i,:, fbands{j})+para,3)));
                    end
                end
            end
        else % average power over each frequency band:
            for i=1:Nchs % for each channel 
                for j=1:length(fbands) % for each frequency band
                    y_range(i,j,:) = mean((S_range(i,:, fbands{j})),3);
                    % ensure y_entire is being created ONCE
                    if isempty(trainY{1,i+2})
                        y_entire(i,j,:) = mean((S_entire(i,:, fbands{j})),3);
                    end
                end
            end
        end
    
        % demean the spectral power data
        for i=1:Nchs % for each channel 
            for j=1:length(fbands) % for each frequency band
                y_range(i,j,:) = y_range(i,j,:) - mean( y_range(i,j,:) );
                % ensure y_entire is being created ONCE
                if isempty(trainY{1,i+2})
                    y_entire(i,j,:) = y_entire(i,j,:) - mean( y_entire(i,j,:) );
                end
            end
        end
        
        % squeeze data together
        for i=1:Nchs/2
            % initiate with first frequency band, for each channel array
            Y_first_ch_range_tmp = [];
            Y_second_ch_range_tmp = [];
    
            % ensure y_entire is being created ONCE
            if isempty(trainY{1,i+2})
                Y_first_ch_entire_tmp = [];
                Y_second_ch_entire_tmp = [];
            end

            % stack array first in order of frequencies then by channel array
            for j=1:length(fbands)
                Y_first_ch_range_tmp = [Y_first_ch_range_tmp, y_range(i,j,:)];
                Y_second_ch_range_tmp = [Y_second_ch_range_tmp, y_range(i+1,j,:)];
                
                % ensure y_entire is being created ONCE
                if isempty(trainY{1,i+2})
                    Y_first_ch_entire_tmp = [Y_first_ch_entire_tmp, y_entire(i,j,:)];
                    Y_second_ch_entire_tmp = [Y_second_ch_entire_tmp, y_entire(i+1,j,:)];
                end
            end
    
            % order the first channel before the second (convention based on
            % original code)
            Y_range_tmp = [Y_first_ch_range_tmp, Y_second_ch_range_tmp];

            % ensure y_entire is being created ONCE
            if isempty(trainY{1,i+2})
                Y_entire_tmp = [Y_first_ch_entire_tmp, Y_second_ch_entire_tmp];
            end

            % squeeze
            trainY{el,i} = squeeze(Y_range_tmp)';
            % ensure y_entire is being created ONCE
            if isempty(trainY{1,i+2})
                trainY{1,i+2} = squeeze(Y_entire_tmp)';
            end
        end

        if train_plot_flag
            % plot spectrogram
            fig = figure('Name',[sprintf('Spectrogram for Trial %d',k)],'NumberTitle','off');
            fig.Position = [720 1440-300*el-1 2000 250];
            plot_matrix(squeeze(S_range(1,:,:)),edges,f); xlabel('');
            hold on; plot(edges, squeeze(y_range(1,5,:)),'w-','LineWidth',2);
            hold on; xline(0,'k-','LineWidth',2);
            hold on; xline(1, 'r-','LineWidth',2);
            ylabel('ACC CH1','fontsize',16);
            xlim([-Tbin Tbin]), set(gca, 'fontsize',16); colorbar('off');
            xlabel('Time (sec)',fontsize=16);
        end
    end
end
end

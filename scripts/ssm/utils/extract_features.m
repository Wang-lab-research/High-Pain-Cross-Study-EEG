function [trainY,fbands,T,tt_entire] = extract_features(Fs,roi_names,events,train_trial,train_trial_id,fft_flag,log_transform_flag,tmin,tmax,crop_flag,train_plot_flag)
%TRAIN_DATALOADER feature extraction 

%% Display
clc; disp('Extracting spectral features...');

%% Main
rng(0); % set seed

Nchs = length(roi_names);

% preallocate
trainY = cell(1,Nchs); % initiate Y matrix (train trial x channels)
onset = events(train_trial_id,1);

if crop_flag
    range = floor(onset-tmin*Fs)+1 : floor(onset+tmax*Fs);
    range
end

% calculate spectral power
for ch=1:Nchs
    
    if crop_flag
        train_trial(ch,:) = train_trial(ch,range);
        size(train_trial)
    end
    
    if fft_flag == 1
        nfft = 2^nextpow2(movingwin(1)*Fs);
        % ACC
        [A11,f11,~] = spectrogram(train_trial(ch,:),movingwin(1)*Fs,(movingwin(1)-movingwin(2))*Fs,nfft,Fs);
        idx = intersect(find(f11>params.fpass(1)),find(f11<params.fpass(2)));
        f_fft = f11(idx)';
        S_temp = abs(A11(idx,:)').^2/nfft/(Fs/2);
        S_range(ch,:,:) = reshape(S_temp,[1,size(S_temp,1),size(S_temp,2)]);

        % S1
        [S11,f11,~] = spectrogram(train_trial(ch+Nchs/2,:),movingwin(1)*Fs,(movingwin(1)-movingwin(2))*Fs,nfft,Fs);
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
        [S_range(ch,:,:),~,~]=mtspecgramc(train_trial(ch,:),movingwin,params);

        [S_entire(ch,:,:),~,~]=mtspecgramc(ACC_raw(ch,:),movingwin,params);

        [S_range(ch+(Nchs/2),:,:),tt_range,~]=mtspecgramc(train_trial(ch+Nchs/2,:),movingwin,params);

        [S_entire(ch+(Nchs/2),:,:),tt_entire,f]=mtspecgramc(S1_raw(ch,:),movingwin,params);
    end

end

% set some more params
edges = tt_range - tmax; % times from mtspecgramc - tmax = 'edges'
T = length(tt_range);
band_theta = find(f>4 & f<8); 
band_alpha = find(f>8 & f<13); 
band_beta = find(f>13 & f<30); 
band_gammaL = find(f>30 & f<58.5);
band_gammaH = find(f>61.5 & f<100); 

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
    xlim([-tmin tmax]), set(gca, 'fontsize',16); colorbar('off');

    subplot(212) % spectrogram of ACC channel 1
    plot_matrix(squeeze(S_range(1,:,:)),edges,f); xlabel('');
    hold on; plot(edges, squeeze(y_range(1,5,:)),'w-','LineWidth',2);
    hold on; xline(0,'k-','LineWidth',2);
    hold on; xline(1, 'r-','LineWidth',2);
    ylabel('ACC CH1','fontsize',16);
    xlim([-tmin tmax]), set(gca, 'fontsize',16); colorbar('off');
    xlabel('Time (sec)',fontsize=16);
end
end

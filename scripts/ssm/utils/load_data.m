function [loaded_data,Fs] = load_data(raw_dir,proc_dir,sub_num,modality_flag)
%LOAD_DATA read data directly from recordings

%% Display
clc; disp("Loading data...");

%% EDF DATA
filePattern = fullfile(raw_dir); % Change to whatever pattern you need.
theFiles = dir(filePattern);
for k = 1:length(theFiles)
    baseFileName = theFiles(k).name;
    if startsWith(baseFileName,sub_num)
        sub_folder = baseFileName;
    end
end

subfilePattern = fullfile([raw_dir,sub_folder]); % Change to whatever pattern you need.
theSubFiles = dir(subfilePattern);
for k = 1:length(theSubFiles)
    baseSubFileName = theSubFiles(k).name;
%     fullFileName = fullfile(theFiles(k).folder, baseFileName);
    if endsWith(baseSubFileName,'.edf')
        acq_id = baseSubFileName;
    end
end

% read directly from original data
[edf_data, edf_label] = edfread([raw_dir,sub_folder,'/',acq_id]);

%% PREPROCESSED DATA, EPOCH TIMES, STIMULUS LABELS, AND DROP LOG

if modality_flag~=1
    proc_path_stem = [proc_dir,sub_num];
else
    proc_path_stem = [proc_dir,sub_folder,'/',sub_num];
end

raw_struct = fiff_setup_read_raw([proc_path_stem,'_preprocessed-raw.fif']);
[raw_data, raw_times] = fiff_read_raw_segment(raw_struct);

% read in epo_times
epo_times = load([proc_path_stem,'_epo_times.mat']);
epo_times = epo_times.epo_times(:,1);

% read in stim_labels
stim_labels = load([proc_path_stem,'_stim_labels.mat']);
stim_labels = stim_labels.stim_labels';

% read in pain_ratings
pain_ratings = load([proc_path_stem,'_pain_ratings.mat']);
pain_ratings = pain_ratings.pain_ratings';

% read in rejected epochs to ignore
drop_log = load([proc_path_stem,'_drop_log.mat']);
drop_log = drop_log.drop_log;

%% Use drop log to modify epo times and stim labels

% use the drop log to remove rejected epochs
epo_times(drop_log) = 0;
stim_labels(drop_log) = 0;
pain_ratings(drop_log) = 42; % cannot use 0 bc pain ratings include 0

% remove zeros from dropped epochs
epo_times = nonzeros(epo_times);
stim_labels = nonzeros(stim_labels);
pain_ratings(pain_ratings==42) = [];

%% Read sampling rate
Fs = raw_struct.info.sfreq;

%% Plot the data and crop if needed, modifying epo_times
close all;
% figure and position
fig = figure('Name','Raw data','NumberTitle','off');
fig.Position = [720 1100 2000 500];

subplot(211); title('Ch 1'); hold on;
plot(1:length(raw_data),raw_data(1,:),'b-','LineWidth',2); hold on;

% event times 
for ii = 1:length(epo_times) % for each stim time 
    hold on; xline(epo_times(ii),'k-','LineWidth',2);
end

ylabel('Potential (uV)','FontSize',16);
set(gca,'FontSize',16);

subplot(212); title('Ch 2'); hold on;
plot(1:length(raw_data),raw_data(2,:),'b-','LineWidth',2); hold on;

% event times 
for ii = 1:length(epo_times) % for each stim time 
    hold on; xline(epo_times(ii),'k-','LineWidth',2);
end

ylabel('Potential (uV)','FontSize',16);
xlabel('Timesteps (a.u.)')
set(gca,'FontSize',16);
sgtitle('RAW DATA','FontSize',18);

%% Crop raw_data to before first stim and after last stim

binSz=5;
raw_data_new = raw_data(:,epo_times(1)-binSz*Fs:epo_times(end)+binSz*Fs);

% get length difference
len_diff = epo_times(1)-binSz*Fs;

% adjust epo_times accordingly
epo_times_new = epo_times - len_diff;

%% Plot CROPPED data

% figure and position
fig = figure('Name','CROPPED data','NumberTitle','off');
fig.Position = [720 200 2000 500];

subplot(211); title('Ch 1'); hold on;
plot(1:length(raw_data_new),raw_data_new(1,:),'b-','LineWidth',2); hold on;

% event times 
for ii = 1:length(epo_times_new) % for each stim time 
    hold on; xline(epo_times_new(ii),'k-','LineWidth',2);
end

ylabel('Potential (uV)','FontSize',16);
set(gca,'FontSize',16);
xlim([0 length(raw_data_new)]);

subplot(212); title('Ch 2'); hold on;
plot(1:length(raw_data_new),raw_data_new(2,:),'b-','LineWidth',2); hold on;

% event times 
for ii = 1:length(epo_times_new) % for each stim time 
    hold on; xline(epo_times_new(ii),'k-','LineWidth',2);
end

ylabel('Potential (uV)','FontSize',16);
xlabel('Timesteps (a.u.)')
set(gca,'FontSize',16);
sgtitle('CROPPED DATA','FontSize',18);
xlim([0 length(raw_data_new)]);


%% Output variables in concise struct

loaded_data.edf_data=edf_data;
loaded_data.edf_label=edf_label;
loaded_data.raw_struct=raw_struct;
loaded_data.raw_data=raw_data_new;
loaded_data.raw_times=raw_times;
loaded_data.epo_times=epo_times_new;
loaded_data.stim_labels=stim_labels;
loaded_data.pain_ratings=pain_ratings;
loaded_data.drop_log=drop_log;

end


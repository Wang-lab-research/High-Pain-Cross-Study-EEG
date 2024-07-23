function [raw_data,selected_goods] = drop_bad_chs(raw_struct,raw_data,selected_chs)
%DROP BAD CHANNELS: use raw_struct.info.bads to remove bad channels
%indentified during preprocessing

%% Display
clc; disp("Dropping bad channels...")

%% Main

ch_names = {'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F11', 'F7', 'F5', 'F3',... 
'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'F12', 'FT11', 'FC5', 'FC3', 'FC1',... 
'Fcz', 'FC2', 'FC4', 'FC6', 'FT12', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2',... 
'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',... 
'TP8', 'M1', 'M2', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',... 
'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Cb1', 'Cb2'};

good_ch_ids = 1:length(ch_names);

% if bads, omit
bads = raw_struct.info.bads;
if ~isempty(bads)
    bads_ids = find(contains(ch_names,bads,IgnoreCase=false));
    raw_data(bads_ids,:) = [];
    good_ch_ids(bads_ids) = [];
end

% new vector for remaining channels
good_ch_names = ch_names(good_ch_ids); % redefine ch_names
selected_goods = find(startsWith(good_ch_names, selected_chs)); % re-select inds

end


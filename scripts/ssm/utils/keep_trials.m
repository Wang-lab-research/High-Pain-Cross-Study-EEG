function [epo_times,stim_labels,pain_ratings,stim_and_ratings,NS_ids,LS_ids,HS_ids] = keep_trials(epo_times,stim_labels,pain_ratings,modality_flag)
%LOAD_DATA read data directly from recordings

%% Display
clc; 

if modality_flag ~=1 % LFP is all hand
    trials_flag = input("Please enter which trials to keep:\n1=Hand only\n2=Back only\n3=Both\n>> ");

    if trials_flag==1
        trials_keep = find(stim_labels<=5);
        NS_ids = find(stim_labels==5);
        LS_ids = find(stim_labels==4);
        HS_ids = find(stim_labels==3);
    elseif trials_flag==2
        trials_keep = find(stim_labels>=6);
        NS_ids = find(stim_labels==8);
        LS_ids = find(stim_labels==7);
        HS_ids = find(stim_labels==6);
    end

else
    trials_keep = 1:length(stim_labels);
    NS_ids = [find(stim_labels==5),find(stim_labels==8)];
    LS_ids = [find(stim_labels==4),find(stim_labels==7)];
    HS_ids = [find(stim_labels==3),find(stim_labels==6)];
end

% adjust arrays accordingly
epo_times = epo_times(trials_keep);
stim_labels = stim_labels(trials_keep);
pain_ratings = pain_ratings(trials_keep);

% combine kept stim labels with pain ratings
stim_and_ratings=zeros(length(stim_labels),2);
stim_and_ratings(:,1) = stim_labels;
stim_and_ratings(:,2) = pain_ratings;

end

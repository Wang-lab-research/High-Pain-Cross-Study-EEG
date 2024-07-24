function [stc_epochs, events, ratings, stimulus_labels] = load_stc_mat(proc_dir, subject_id)
%LOAD_STC_MAT Load STC data processed in Python.

abs_proc_dir = fullfile(pwd, proc_dir);
addpath(genpath(abs_proc_dir));
stc_epochs_dir = fullfile(abs_proc_dir, subject_id + "_stc_epochs/");
events_path = fullfile(abs_proc_dir, subject_id + "_events.mat");
pain_ratings_path = fullfile(abs_proc_dir, subject_id + "_pain_ratings.mat");
stimulus_labels_path = fullfile(abs_proc_dir, subject_id + "_stimulus_labels.mat");

mat_files = dir(fullfile(stc_epochs_dir, '*.mat'));

% Determine the dimensions of the data in the first file to initialize the 3D matrix
first_file_path = fullfile(stc_epochs_dir, mat_files(1).name);
first_data = load(first_file_path);
data_dim = size(first_data.data);

% Initialize the 3D matrix to hold all the data
num_files = length(mat_files);
stc_epochs = zeros([data_dim(1), num_files, data_dim(2)]);

% Load each MAT file into the 3D matrix
for i = 1:num_files
    file_name = mat_files(i).name;
    file_path = fullfile(stc_epochs_dir, file_name);
    data = load(file_path);
    stc_epochs(:, i, :) = data.data; % Store data in the i-th slice of the 3D matrix
end

events = load(events_path).data;
ratings = load(pain_ratings_path).data';
stimulus_labels = load(stimulus_labels_path).data';

end


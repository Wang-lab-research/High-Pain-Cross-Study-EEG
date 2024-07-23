% Initialize variables to store results
train_trial_ids = [];
precisions = [];
recalls = [];
f1s = [];
confusion_matrices = [];
aucs = [];
    
for i = 1:folds
    % Randomly sample an event ID at high pain threshold or above
    high_pain_indices = find(ratings > high_pain_threshold);
    train_trial_id = randsample(high_pain_indices, 1);
    train_trial_ids = [train_trial_ids; train_trial_id];
    
    train_trial = epochs(train_trial_id);
    test_trials = epochs;
    test_trials(train_trial_id) = [];
    test_trial_ids = 1:length(epochs);
    test_trial_ids(train_trial_id) = [];
    test_labels = ratings;
    test_labels(train_trial_id) = [];


    % Initialize variables to store results
    avg_precisions = [];
    avg_recalls = [];
    avg_f1s = [];
    avg_confusion_matrices = [];
    avg_aucs = [];

    for j = 1:length(test_trials)
        % Train model (Placeholder: replace with your model training code)
        % model = trainModel(train_trial, ...);

        % Test model (Placeholder: replace with your model testing code)
        % predicted_labels = testModel(model, test_trials{j});

        % For illustration, let's assume predicted_labels are generated
        % randomly (replace with actual model predictions)
        predicted_labels = randi([0, 1], size(test_labels));

        % Calculate precision, recall, f1, confusion matrix, and ROC AUC
        [precision, recall, f1, ~] = precision_recall_fscore_support(test_labels, predicted_labels, 'average', 'binary');
        confusion_matrix = confusionmat(test_labels, predicted_labels);
        auc = roc_auc_score(test_labels, predicted_labels);

        % Store results
        avg_precisions = [avg_precisions; precision];
        avg_recalls = [avg_recalls; recall];
        avg_f1s = [avg_f1s; f1];
        avg_confusion_matrices = [avg_confusion_matrices; confusion_matrix(:)];
        avg_aucs = [avg_aucs; auc];
    end

    % Calculate average precision, recall, f1, confusion matrix, and ROC AUC
    avg_precision = mean(avg_precisions);
    avg_recall = mean(avg_recalls);
    avg_f1 = mean(avg_f1s);
    avg_confusion_matrix = reshape(mean(avg_confusion_matrices, 1), 2, 2);
    avg_auc = mean(avg_aucs);

    % Store results for each fold to choose best model from training trials options
    precisions = [precisions; avg_precision];
    recalls = [recalls; avg_recall];
    f1s = [f1s; avg_f1];
    confusion_matrices = cat(3, confusion_matrices, avg_confusion_matrix);
    aucs = [aucs; avg_auc];
end
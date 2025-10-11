function run()
    clc; clear; close all;
    
    % Use relative path - assumes data folder is in same directory as script
    dataset_folder = './data_mat';
    output_csv = 'batch_results.csv';
    
    % Parameters

    r_values = 5;
    lambda_values = 0.1;
    alpha_values = 0.6;
    fold_setting = 1;
    % Get all .mat files in the dataset folder
    % mat_files = dir(fullfile(dataset_folder, '*.mat'));
    mat_files = dir(fullfile(dataset_folder, 'Mfeat_m_5_825.mat'));
    results_table = table();
    



    for i = 1:length(mat_files)
        dataset_name = mat_files(i).name;
        dataset_path = fullfile(dataset_folder, dataset_name);
        
        try
            all_results = BGFMOD(dataset_path, fold_setting, r_values, lambda_values, alpha_values);
            [best_auc, best_params, best_recon, best_cons] = evaluate_all_combinations(all_results);
            new_row = table();
            new_row.Dataset = {strrep(dataset_name, '.mat', '')};
            new_row.Best_AUC = best_auc;
            new_row.Best_r = best_params(1);
            new_row.Best_lambda = best_params(2);
            new_row.Best_alpha = best_params(3);
            new_row.Best_recon_score = best_recon;
            new_row.Best_cons_score = best_cons;
            results_table = [results_table; new_row];
        catch ME
            % Handle failed cases with NaN values
            new_row = table();
            new_row.Dataset = {strrep(dataset_name, '.mat', '')};
            new_row.Best_AUC = NaN;
            new_row.Best_r = NaN;
            new_row.Best_lambda = NaN;
            new_row.Best_alpha = NaN;
            new_row.Best_recon_score = NaN;
            new_row.Best_cons_score = NaN;
            
            results_table = [results_table; new_row];
        end
    end
    
    % Save results to CSV
    writetable(results_table, output_csv);
    
    % Display summary
    successful_runs = sum(~isnan(results_table.Best_AUC));
    fprintf('Processing completed. %d/%d datasets processed successfully.\n', successful_runs, length(mat_files));
    
    if successful_runs > 0
        [~, best_idx] = max(results_table.Best_AUC);
        fprintf('Best performing dataset: %s (AUC: %.4f)\n', ...
            results_table.Dataset{best_idx}, results_table.Best_AUC(best_idx));
    end
end

function [best_auc, best_params, best_recon, best_cons] = evaluate_all_combinations(all_results)
    best_auc = -inf;
    best_params = [0, 0, 0];
    best_recon = 0;
    best_cons = 0;
    
    for i = 1:length(all_results)
        result = all_results{i};
        recon_scores = result.recon_scores;
        cons_scores = result.cons_scores;
        labels = result.labels;
        alpha = result.alpha;
        
        % Normalize reconstruction scores
        recon_min = min(recon_scores);
        recon_max = max(recon_scores);
        if recon_max > recon_min
            recon_norm = (recon_scores - recon_min) / (recon_max - recon_min);
        else
            recon_norm = zeros(size(recon_scores));
        end
        
        % Normalize consistency scores
        cons_min = min(cons_scores);
        cons_max = max(cons_scores);
        if cons_max > cons_min
            cons_norm = (cons_scores - cons_min) / (cons_max - cons_min);
        else
            cons_norm = zeros(size(cons_scores));
        end
        
        % Compute final scores
        final_scores = alpha * recon_norm + (1 - alpha) * cons_norm;
        
        try
            [~, ~, ~, current_auc] = perfcurve(labels, final_scores, 1);
        catch
            current_auc = NaN;
        end
        
        if ~isnan(current_auc) && current_auc > best_auc
            best_auc = current_auc;
            best_params = [result.r, result.lambda, result.alpha];
            best_recon = mean(recon_scores);
            best_cons = mean(cons_scores);
        end
    end
end
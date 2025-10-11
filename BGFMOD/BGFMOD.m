function all_results = BGFMOD(data_path, fold_setting, r_values, lambda_values, alpha_values)

    ds = load(data_path);
    

    all_folds = 1:length(ds.out_label);
    if ischar(fold_setting) && strcmpi(fold_setting, 'all')
        fold_list = all_folds;
    elseif isnumeric(fold_setting)
        fold_list = fold_setting(:).';
    end
    n_folds_selected = numel(fold_list);

    [X_all, labels_all, S_all, Scons_all, info_all, Sxsum_all, Amasks_all] = ...
        precompute_fold_data(data_path, fold_list, n_folds_selected);
    

    min_n = min(cellfun(@(ii) ii.n_samples, info_all));
    r_values = r_values(r_values >= 1 & r_values <= max(1, min_n-1));

    params = initialize_algorithm_parameters();
    

    all_results = perform_grid_search(X_all, labels_all, S_all, Scons_all, ...
        info_all, Sxsum_all, Amasks_all, r_values, lambda_values, alpha_values, ...
        n_folds_selected, params);
end

function [X_all, labels_all, S_all, Scons_all, info_all, Sxsum_all, Amasks_all] = ...
    precompute_fold_data(data_path, fold_list, n_folds_selected)

    X_all = cell(n_folds_selected, 1);
    labels_all = cell(n_folds_selected, 1);
    S_all = cell(n_folds_selected, 1);
    Scons_all = cell(n_folds_selected, 1);
    info_all = cell(n_folds_selected, 1);
    Sxsum_all = cell(n_folds_selected, 1);
    Amasks_all = cell(n_folds_selected, 1);
    
    for k = 1:n_folds_selected
        f = fold_list(k);
        [X_views, labels, info] = data_loader_fixed(data_path, f);
        [S_views, S_consensus] = compute_similarity_matrices_fixed(X_views);

        Sx_sum = zeros(info.n_samples, info.n_samples);
        for v = 1:length(X_views)
            Xv = X_views{v};
            Sx_sum = Sx_sum + (Xv * Xv');
        end
        
        A_masks = build_A_masks_knn(S_views);

        X_all{k} = X_views;
        labels_all{k} = labels;
        S_all{k} = S_views;
        Scons_all{k} = S_consensus;
        info_all{k} = info;
        Sxsum_all{k} = Sx_sum;
        Amasks_all{k} = A_masks;
    end
end

function params = initialize_algorithm_parameters()
% Initialize optimization parameters
    params = struct();
    params.max_outer_iter = 25;
    params.tolerance = 1e-6;
    params.rho = 0.5;
    params.inner_iter = 15;
    params.inner_sub_iter = 2;
    % params.rel_change_tol = 1e-4;
end

function all_results = perform_grid_search(X_all, labels_all, S_all, Scons_all, ...
    info_all, Sxsum_all, Amasks_all, r_values, lambda_values, alpha_values, ...
    n_folds_selected, params)
% Perform grid search over all parameter combinations
    
    numR = numel(r_values);
    numL = numel(lambda_values);
    numA = numel(alpha_values);
    all_results = {};
    
    for ri = 1:numR
        r_val = r_values(ri);
        for li = 1:numL
            lam_val = lambda_values(li);
            for ai = 1:numA
                alpha_val = alpha_values(ai);
                
                % Process all folds for current parameter combination
                [fold_recon_scores, fold_cons_scores] = process_folds(...
                    X_all, S_all, Scons_all, info_all, Sxsum_all, Amasks_all, ...
                    r_val, lam_val, n_folds_selected, params);
                
                % Aggregate results across folds
                current_result = aggregate_fold_results(fold_recon_scores, ...
                    fold_cons_scores, labels_all, r_val, lam_val, alpha_val, n_folds_selected);
                
                all_results{end+1} = current_result;
            end
        end
    end
end

function [fold_recon_scores, fold_cons_scores] = process_folds(X_all, S_all, ...
    Scons_all, info_all, Sxsum_all, Amasks_all, r_val, lam_val, n_folds_selected, params)
% Process all folds for a given parameter combination
    
    fold_recon_scores = cell(n_folds_selected, 1);
    fold_cons_scores = cell(n_folds_selected, 1);
    
    for k = 1:n_folds_selected
        X_views = X_all{k};
        S_views = S_all{k};
        S_consensus = Scons_all{k};
        Sx_sum = Sxsum_all{k};
        A_masks = Amasks_all{k};
        
        % Initialize consensus subspace
        U_init = initialize_consensus_subspace_fixed(S_consensus, r_val);
        
        % Optimization
        U_final = optimize_subspace(X_views, S_views, A_masks, U_init, ...
            Sx_sum, lam_val, r_val, params);
        
        % Compute anomaly scores
        [recon_scores, cons_scores] = compute_anomaly_scores_components(...
            X_views, S_views, U_final, A_masks);
        
        fold_recon_scores{k} = recon_scores;
        fold_cons_scores{k} = cons_scores;
    end
end

function U_final = optimize_subspace(X_views, S_views, A_masks, U_init, ...
    Sx_sum, lam_val, r_val, params)
% Main optimization routine for subspace learning
    
    B_att = lam_val * Sx_sum;
    U_old = U_init;
    prev_F = inf;
    
    for iter = 1:params.max_outer_iter
        w_cat = 1 - lam_val;
        
        % Inner optimization step
        U_new = inner_optimization_optimized(B_att, A_masks, U_old, ...
            length(X_views), lam_val, r_val, w_cat);
        
        % Check convergence
        rel_err = norm(U_new - U_old, 'fro') / (norm(U_old, 'fro') + eps);
        F_total = compute_objective_optimized(X_views, S_views, U_new, ...
            A_masks, lam_val, length(X_views));
        
        if ~isinf(prev_F)
            rel_drop = (F_total - prev_F) / (abs(prev_F) + eps);
        end
        prev_F = F_total;
        U_old = U_new;
        
        if rel_err < params.tolerance
            break;
        end
    end
    
    U_final = U_new;
end

function current_result = aggregate_fold_results(fold_recon_scores, ...
    fold_cons_scores, labels_all, r_val, lam_val, alpha_val, n_folds_selected)
% Aggregate results across all folds
    
    current_result = struct();
    current_result.r = r_val;
    current_result.lambda = lam_val;
    current_result.alpha = alpha_val;
    current_result.recon_scores = [];
    current_result.cons_scores = [];
    current_result.labels = [];
    
    for k = 1:n_folds_selected
        current_result.recon_scores = [current_result.recon_scores; fold_recon_scores{k}];
        current_result.cons_scores = [current_result.cons_scores; fold_cons_scores{k}];
        current_result.labels = [current_result.labels; labels_all{k}];
    end
end

function [recon_scores, cons_scores] = compute_anomaly_scores_components(X_views, S_views, U, A_masks)
% Compute reconstruction and consistency anomaly scores
    
    n_views = numel(X_views);
    n = size(X_views{1}, 1);
    
    % Reconstruction scores
    recon_each = zeros(n, n_views);
    for v = 1:n_views
        Xv = X_views{v};
        R = Xv - U*(U'*Xv);
        d_v = size(Xv, 2);
        row_energy = sum(R.^2, 2);
        scale_v = norm(Xv, 'fro')^2 / max(d_v, 1);
        recon_each(:, v) = row_energy / (scale_v + eps);
        % recon_each(:, v) = row_energy ./ sum(Xv.^2, 2);
    end
    recon_each = robust_normalize_cols(recon_each);
    recon_scores = mean(recon_each, 2);
    % recon_scores = max(recon_each, [], 2);

    % Consistency scores
    cons_vec = zeros(n, 1);
    num_pairs = 0;
    P = eye(n) - U*U';
    
    for v = 1:n_views
        for q = v+1:n_views
            A = S_views{v} - S_views{q};
            if ~isempty(A_masks{v,q})
                A = A_masks{v,q};
            else
                A = S_views{v} - S_views{q};
                nf = norm(A, 'fro');
                if nf > 0
                    A = A / (nf + 1e-12);
                end
            end
            
            UTA = U' * A;
            AU = A * U;
            UTAU = U' * AU;
            Z = A - U*(UTA) - AU*U' + U*(UTAU)*U';
            Z = 0.5*(Z + Z');
            row_energy = sum(Z.^2, 2) / max(n, 1);
            cons_vec = cons_vec + row_energy;
            num_pairs = num_pairs + 1;
        end
    end
    cons_scores = cons_vec / max(num_pairs, 1);
end

function A_masks = build_A_masks_knn(S_views)
% Build adaptive masks based on k-nearest neighbors
    
    n_views = numel(S_views);
    n_samples = size(S_views{1}, 1);
    k_neighbors = max(min(round(0.05*n_samples), 50), 5);
    
    % Find top-k neighbors for each view
    topk = cell(n_views, 1);
    for v = 1:n_views
        Sv = S_views{v};
        Sv(1:n_samples+1:end) = 0; % Remove diagonal
        [~, topk{v}] = maxk(Sv, min(k_neighbors, n_samples-1), 2);
    end
    
    % Build masks for each view pair
    A_masks = cell(n_views, n_views);

total_pairs = 0;

for v = 1:n_views
    for q = v+1:n_views
        total_pairs = total_pairs + 1;

        M = sparse(n_samples, n_samples);
        
        Nk_v = topk{v};
        Nk_q = topk{q};
        

        for i = 1:n_samples
            Nv = Nk_v(i,:);
            Nq = Nk_q(i,:);
            Nij = intersect(Nv, Nq);
            
            if isempty(Nij)
                Nij = union(Nv, Nq);
            end
            
            Nij = Nij(Nij > 0 & Nij <= n_samples);
            
            if ~isempty(Nij)
                M(i, Nij) = true;
            end
        end
        
        M = M | M';

        A = sparse(S_views{v} - S_views{q});
        A = A .* M;
        A = 0.5 * (A + A');
        
        nf = norm(A, 'fro');
        if nf > 0
            A = A / (nf + 1e-12);
        end
        
        A_masks{v,q} = A;
        


        
        clear M Nk_v Nk_q;
    end
end

end

function [X_views, labels, data_info] = data_loader_fixed(data_path, fold_idx)
% Load and preprocess data for a specific fold
    
    data_struct = load(data_path);
    X_raw = data_struct.X; 
    out_label_raw = data_struct.out_label;
    
    X_views = cell(1, length(X_raw{fold_idx}));
    % 
    for j = 1:length(X_raw{fold_idx})
        X_views{j} = X_raw{fold_idx}{j}.';
    end  

    labels = out_label_raw{fold_idx}(:);
    
    % Normalize features
    for v = 1:length(X_views)
        X_views{v} = NormalizeFea(X_views{v}, 0);
    end
    
    % Store data information
    data_info = struct();
    data_info.n_views = length(X_views);
    data_info.n_samples = size(X_views{1}, 1);
    data_info.view_dims = cellfun(@(x) size(x, 2), X_views);
    
    validate_data_consistency_fixed(X_views, labels);
end

function validate_data_consistency_fixed(X_views, labels)
% Validate data consistency across views
    
    n_views = length(X_views);
    n_samples = length(labels);
    
    for v = 1:n_views
        assert(size(X_views{v}, 1) == n_samples, ...
            'Inconsistent sample count between view %d and labels', v);
    end
end

function [S_views, S_consensus] = compute_similarity_matrices_fixed(X_views)
% Compute similarity matrices for all views and consensus matrix
    
    n_views = length(X_views);
    n_samples = size(X_views{1}, 1);
    S_views = cell(n_views, 1);
    
    % Compute view-specific similarity matrices
    for v = 1:n_views
        Xv = X_views{v};
        sigma = estimate_gaussian_bandwidth_fixed(Xv);
        S_views{v} = compute_gaussian_similarity_fixed(Xv, sigma);
    end
    
    % Compute consensus similarity matrix
    S_consensus = zeros(n_samples, n_samples);
    for v = 1:n_views
        S_consensus = S_consensus + S_views{v};
    end
    S_consensus = S_consensus / n_views;
    S_consensus = 0.5 * (S_consensus + S_consensus');
    S_consensus = S_consensus / max(S_consensus(:) + eps);
end

function sigma = estimate_gaussian_bandwidth_fixed(X)
% Estimate Gaussian bandwidth using median heuristic
    
    distances = pdist(X, 'euclidean');
    sigma = median(distances);
    if sigma == 0 || isnan(sigma)
        sigma = 1.0;
    end
end

function S = compute_gaussian_similarity_fixed(X, sigma)
% Compute Gaussian similarity matrix
    
    n = size(X, 1);
    S = zeros(n, n);
    
    for i = 1:n
        xi = X(i, :);
        for j = i:n
            xj = X(j, :);
            dist_sq = sum((xi - xj).^2);
            val = exp(-dist_sq / (2 * sigma^2));
            S(i, j) = val; 
            S(j, i) = val;
        end
    end
    
    mx = max(S(:));
    if mx > 0
        S = S / mx;
    end
end

function U_init = initialize_consensus_subspace_fixed(S_consensus, r)
% Initialize consensus subspace using eigendecomposition
    
    S_consensus = 0.5 * (S_consensus + S_consensus');
    S_consensus = S_consensus + 1e-8 * eye(size(S_consensus));
    
    [V, Lambda] = eig(S_consensus);
    [~, idx] = sort(diag(Lambda), 'descend');
    V = V(:, idx);
    
    r = max(1, min(r, size(S_consensus, 1)));
    [U_init, ~] = qr(V(:, 1:r), 0);
    U_init = U_init(:, 1:r);
end

function F_total = compute_objective_optimized(X_views, S_views, U, A_masks, lambda_values, n_views)
% Compute the total objective function value
    
    n = size(U, 1);
    P = eye(n) - U * U';
    
    % Consistency term
    F_cat = 0;
    for v = 1:n_views
        for q = v+1:n_views
            if ~isempty(A_masks{v,q})
                Z = P * A_masks{v,q} * P;
                row_norms = sum(Z.^2, 2);
                F_cat = F_cat + sum(row_norms);
            end
        end
    end
    
    % Reconstruction term
    F_att = 0;
    for v = 1:n_views
        Xv = X_views{v};
        Xv_F_norm = norm(Xv, 'fro');
        Xv_normalized = Xv / (Xv_F_norm + eps);
        Z = P * Xv_normalized;
        row_norms = sum(Z.^2, 2);
        F_att = F_att + sum(row_norms);
    end
    
    F_total = (1-lambda_values)*F_cat + lambda_values * F_att;
end

function Xn = robust_normalize_cols(X)
% Robust column-wise normalization using median and MAD
    
    Xn = zeros(size(X));
    for j = 1:size(X, 2)
        x = X(:, j);
        medv = median(x);
        madv = mad(x, 1);
        
        if madv < eps
            y = x - medv;
        else
            y = (x - medv) ./ (1.4826*madv);
        end
        
        q1 = prctile(y, 1); 
        q99 = prctile(y, 99);
        y = min(max(y, q1), q99);
        y = (y - min(y)) / (max(y) - min(y) + eps);
        Xn(:, j) = y;
    end
end

function U_new = inner_optimization_optimized(B_att, A_masks, U_current, n_views, lam_val, r, w_cat)
% Inner optimization routine using eigendecomposition
    
    n_samples = size(U_current, 1);
    Q_t = eye(n_samples) - U_current * U_current';
    B = B_att;
    
    % Add consistency terms
    for v = 1:n_views
        for q = v+1:n_views
            if ~isempty(A_masks{v,q})
                AQA = A_masks{v,q}' * Q_t * A_masks{v,q};
                B = B + w_cat * AQA;
            end
        end
    end
    
    B = B + 1e-8 * speye(n_samples);
    B = 0.5*(B + B') + 1e-8*speye(n_samples);
    
    % First eigendecomposition
    opts.tol = 1e-6; 
    opts.maxit = 300;
    try
        [V, ~] = eigs(B, r, 'largestreal', opts);
    catch
        [Vfull, Lfull] = eig(full(B));
        [~, idx] = sort(diag(Lfull), 'descend');
        V = Vfull(:, idx(1:r));
    end
    
    [U_t1, ~] = qr(V, 0);
    U_t1 = U_t1(:, 1:r);
    
    % Second optimization step
    P_t1 = eye(n_samples) - U_t1 * U_t1';
    B2 = B_att;
    
    for v = 1:n_views
        for q = v+1:n_views
            if ~isempty(A_masks{v,q})
                APA = A_masks{v,q}' * P_t1 * A_masks{v,q};
                B2 = B2 + w_cat * APA;
            end
        end
    end
    
    B2 = B2 + 1e-8 * speye(n_samples);
    B2 = 0.5*(B2 + B2') + 1e-8*speye(n_samples);
    
    try
        [V2, ~] = eigs(B2, r, 'largestreal', opts);
    catch
        [Vfull2, Lfull2] = eig(full(B2));
        [~, idx2] = sort(diag(Lfull2), 'descend');
        V2 = Vfull2(:, idx2(1:r));
    end
    
    [U_new, ~] = qr(V2, 0);
    U_new = U_new(:, 1:r);
end

function X_normalized = NormalizeFea(X, type)
% Feature normalization with different methods
%
% Inputs:
%   X    - Input feature matrix type - Normalization type: 0 for z-score, 1
%   for min-max
%
% Output:
%   X_normalized - Normalized feature matrix

    if nargin < 2
        type = 0;
    end
    
    switch type
        case 0  % Z-score normalization
            X_normalized = (X - mean(X, 1)) ./ (std(X, 0, 1) + eps);
        case 1  % Min-max normalization
            X_normalized = (X - min(X, [], 1)) ./ (max(X, [], 1) - min(X, [], 1) + eps);
        otherwise
            X_normalized = X;
    end
end
function inloc_compute_scores(varargin)
    % Config fields:
    % input_db_features_mat_path, input_query_features_mat_path
    % output_scores_mat_path
    % Computes similarity scores for query images with database images.
    % Implementation reworked from the old buildFileLists/buildScores.m.

    [filepath, ~, ~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', '..', 'functions', 'inLocCIIRC_utils'));
    inloc_add_abs_fn_path('yaml');

    params = inloc_parse_inputs(varargin{:}).scores;

    db_features = load(params.input_db_features_mat_path).features;
    query_features = load(params.input_query_features_mat_path).features;
    n_img = size(db_features, 2);
    n_query = size(query_features, 2);
    coarse_feature_level = get_with_default(params, 'input_feature_layer', 5);
    score = struct('query_path', {}, 'scores', {}, 'db_score_paths', {});

    db_paths = {};
    all_db_features = zeros(n_img, size(db_features(1).features{coarse_feature_level}.x(:), 1));
    for i=1:n_img
        all_db_features(i, :) = db_features(i).features{coarse_feature_level}.x(:)';
        db_paths{i} = db_features(i).img_path;
    end
    all_db_features = all_db_features';
    all_db_features = all_db_features ./ vecnorm(all_db_features);
    check_is_normalized(all_db_features);

    for i=1:n_query
        fprintf('processing query %d/%d\n', i, n_query);
        single_q_features = query_features(i).features{coarse_feature_level}.x(:)';
        single_q_features = single_q_features ./ vecnorm(single_q_features);
        check_is_normalized(single_q_features);
        single_q_features = repmat(single_q_features, n_img, 1)';
        similarityScores = dot(single_q_features, all_db_features);
        score(i).query_path = query_features(i).img_path;
        score(i).scores = single(similarityScores); % NOTE: this is not a probability distribution (and it does not have to be)
        score(i).db_score_paths = db_paths;
    end

    create_parent_folder(params.output_scores_mat_path);
    save(params.output_scores_mat_path, 'score');
end

function check_is_normalized(feat)
    tol = 1e-6;
    if ~all(abs(vecnorm(feat) - 1.0) < tol)
        fprintf('norm: %f\n', vecnorm(feat));
        error('Features are not normalized!');
    end
end

function buildScores(varargin)

    addpath(absFnPath('InLocCIIRC_utils', 'params'));
    addpath(absFnPath('yaml'));

    parser = inputParser;
    addOptional(parser, 'config', "use-old-implementation");
    addParameter(parser, 'params', struct());
    parse(parser, varargin{:});

    if parser.Results.config ~= "use-old-implementation"
        params = ReadYaml(parser.Results.config);

        [output_dir, ~, ~] = fileparts(params.scores.output_scores_mat_path);
        if exist(output_dir, 'dir') ~= 7
            mkdir(output_dir);
        end

        load(params.scores.input_features_mat_path, 'query_features', 'db_features');
        n_img = size(db_features, 2);
        n_query = size(query_features, 2);
        score = struct('query_name', {}, 'scores', {});

        all_db_features = zeros(n_img, size(db_features(1).features, 1));
        for i=1:n_img
            all_db_features(i, :) = db_features(i).features';
        end
        all_db_features = all_db_features';

        tol = 1e-6;
        if ~all(abs(vecnorm(all_db_features)-1.0)<tol)
            fprintf('norm: %f\n', vecnorm(all_db_features));
            error('Features are not normalized!');
        end

        for i=1:n_query
            fprintf('processing query %d/%d\n', i, n_query);
            single_q_features = query_features(i).features';
            if ~all(abs(norm(single_q_features)-1.0)<tol)
                fprintf('norm: %f\n', norm(single_q_features));
                error('Features are not normalized!');
            end
            single_q_features = repmat(single_q_features, n_img, 1)';
            similarityScores = dot(single_q_features, all_db_features);
            score(i).query_name = query_features(i).query_name;
            score(i).scores = single(softmax(similarityScores)); % NOTE: this is not a probability distribution (and it does not have to be)
        end

        save(params.scores.output_scores_mat_path, 'score');

    else
        featuresPath = fullfile(params.input.feature.dir, 'computed_features.mat');

        files = dir(fullfile(params.dataset.db.cutouts.dir, '**/cutout*.jpg'));
        nCutouts = size(files,1); % TODO: use the actual number from featuresPath

        %x = matfile(featuresPath);
        load(featuresPath, 'queryFeatures', 'cutoutFeatures');
        nQueries = size(queryFeatures,2);
        score = struct('queryname', {}, 'scores', {});

        allCutoutFeatures = zeros(nCutouts, size(cutoutFeatures(1).features,1));
        for i=1:nCutouts
            allCutoutFeatures(i,:) = cutoutFeatures(i).features';
        end
        allCutoutFeatures = allCutoutFeatures';

        tol = 1e-6;
        if ~all(abs(vecnorm(allCutoutFeatures)-1.0)<tol)
            fprintf('norm: %f\n', vecnorm(allCutoutFeatures));
            error('Features are not normalized!');
        end
        for i=1:nQueries
            fprintf('processing query %d/%d\n', i, nQueries);
            thisQueryFeatures = queryFeatures(i).features';
            if ~all(abs(norm(thisQueryFeatures)-1.0)<tol)
                fprintf('norm: %f\n', norm(thisQueryFeatures));
                error('Features are not normalized!');
            end
            thisQueryFeatures = repmat(thisQueryFeatures, nCutouts, 1)';
            similarityScores = dot(thisQueryFeatures, allCutoutFeatures);
            score(i).queryname = queryFeatures(i).queryname;
            score(i).scores = single(similarityScores); % NOTE: this is not a probability distribution (and it does not have to be)
        end

        save(params.input.scores.path, 'score');
    end
end

function [path] = absFnPath(varargin)
    % absFnPath Get valid absolute OS filesystem path of specified function name.
    % Relies on the file structure of this repository.
    
    % Get absolute path to folder containing this very file.
    [filepath, ~, ~] = fileparts(mfilename('fullpath'));
    % OS independent path creation.
    path = fullfile(filepath, '..', 'functions', varargin{:});
end

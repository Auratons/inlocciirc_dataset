function inloc_compute_features(varargin)
    % Config fields:
    % input_netvlad_pretrained
    % input_db_mat_path, input_query_mat_path
    % output_db_features_mat_path, output_query_features_mat_path
    % Computes features for files in filelists and saves them for later processing.
    % Implementation reworked from the old buildScores/buildFeatures.m.

    [filepath, ~, ~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', '..', 'functions', 'inLocCIIRC_utils'));
    inloc_add_abs_fn_path('inLocCIIRC_utils', 'at_netvlad_function');
    inloc_add_abs_fn_path('relja_matlab');
    inloc_add_abs_fn_path('relja_matlab', 'matconvnet');
    inloc_add_abs_fn_path('netvlad');
    inloc_add_abs_fn_path('yaml');
    run(fullfile(filepath, '..', '..', 'functions', 'matconvnet', 'matlab', 'vl_setupnn.m'));

    params = inloc_parse_inputs(varargin{:}).features;

    featureLength = 32768;

    load(params.input_netvlad_pretrained, 'net');
    net = relja_simplenn_tidy(net);
    net = relja_cropToLayer(net, 'postL2');

    types = {'db', 'query'};
    for idx = 1:length(types)
        type = types{idx};
        output_mat_path = params.(sprintf('output_%s_features_mat_path', type));

        if exist(output_mat_path, 'file') ~= 2
            load(params.(sprintf('input_%s_mat_path', type)), 'filenames');
            file_count = size(filenames, 2);
            features = struct('img_path', {}, 'features', {});
            for i=1:file_count
                fprintf('Finding features for %s image #%d/%d\n\n', type, i, file_count)
                img_path = filenames{i};
                img = imread(img_path);
                cnn = at_serialAllFeats_convfeat(net, img, 'useGPU', true);
                for l = [1 2 4 5]
                    cnn{l} = [];
                end
                features(i).img_path = img_path;
                features(i).features = cnn;
            end

            create_parent_folder(output_mat_path);
            save(output_mat_path, 'features', '-v7.3');
        else
            fprintf('SKIPPING FEATURE COMPUTATION, output "%s" already exists.\n', output_mat_path);
        end
    end
end

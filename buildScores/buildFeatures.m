function buildFeatures(varargin)

    addpath(absFnPath('relja_matlab'));
    addpath(absFnPath('relja_matlab', 'matconvnet'));
    addpath(absFnPath('netvlad'));
    addpath(absFnPath('InLocCIIRC_utils', 'at_netvlad_function'));
    addpath(absFnPath('InLocCIIRC_utils', 'params'));
    addpath(absFnPath('yaml'));
    run(absFnPath('matconvnet', 'matlab', 'vl_setupnn.m'));

    parser = inputParser;
    addOptional(parser, 'config', "use-old-implementation");
    addOptional(parser, 'config_section', "");
    addParameter(parser, 'params', struct());
    parse(parser, varargin{:});

    featureLength = 32768;

    if parser.Results.config ~= "use-old-implementation"
        % Config fields:
        % input_netvlad_pretrained
        % input_db_mat_path
        % input_query_mat_path
        % output_features_mat_path
        params = ReadYaml(parser.Results.config);
        if parser.Results.config_section ~= ""
            params = params.(parser.Results.config_section);
        end

        [output_dir, ~, ~] = fileparts(params.features.output_features_mat_path);
        if exist(output_dir, 'dir') ~= 7
            mkdir(output_dir);
        end

        load(params.features.input_netvlad_pretrained, 'net');
        net = relja_simplenn_tidy(net);
        net = relja_cropToLayer(net, 'postL2');

        % DB images
        db_imgnames_all = load(params.features.input_db_mat_path);
        db_imgnames_all = db_imgnames_all.db_imgnames_all;
        dbSize = size(imread(db_imgnames_all{1}));
        dbSize = [dbSize(2), dbSize(1)]; % width, height
        n_img = size(db_imgnames_all, 2);
        db_features = struct('db_img_name', {}, 'features', {});
        for i=1:110 %n_img
            fprintf('Finding features for db image #%d/%d\n\n', i, n_img)
            db_img_name = db_imgnames_all{i};
            img = imread(db_img_name);  % sqresize(imread(db_img_name), 512);
            cnn = at_serialAllFeats_convfeat(net, img, 'useGPU', true);
            for l = [1 2 4 6] cnn{l} = []; end
            db_features(i).db_img_name = db_img_name;
            % db_features(i).features = cnn{6}.x(:);
            db_features(i).features = cnn;
        end

        % Query images
        query_imgnames_all = load(params.features.input_query_mat_path);
        query_imgnames_all = query_imgnames_all.query_imgnames_all;
        n_query = size(query_imgnames_all,2);
        query_features = struct('query_name', {}, 'features', {});
        for i=1:50 %n_query
            fprintf('Finding features for query #%d/%d\n\n', i, n_query)
            query_name = query_imgnames_all{i};
            img = imread(query_name);  % sqresize(imread(query_name), 512);
            cnn = at_serialAllFeats_convfeat(net, img, 'useGPU', true);
            for l = [1 2 4 6] cnn{l} = []; end
            query_features(i).query_name = query_name;
            % query_features(i).features = cnn{6}.x(:);
            query_features(i).features = cnn;
        end

        %% save the features
        [output_dir, ~, ~] = fileparts(params.features.output_features_mat_path);
        if exist(output_dir, 'dir') ~= 7
            mkdir(output_dir);
        end
        save(params.features.output_features_mat_path, 'query_features', 'db_features', '-v7.3');

    else
        params = parser.Results.params;
        queryDirWithSlash = [params.dataset.query.dir, '/'];

        x = load(params.input.dblist.path);
        cutoutImageFilenames = x.cutout_imgnames_all;
        cutoutSize = size(imread(fullfile(params.dataset.db.cutouts.dir, cutoutImageFilenames{1})));
        cutoutSize = [cutoutSize(2), cutoutSize(1)]; % width, height

        if exist(params.input.feature.dir, 'dir') ~= 7
            mkdir(params.input.feature.dir);
        end

        load(params.netvlad.dataset.pretrained, 'net');
        net = relja_simplenn_tidy(net);
        net = relja_cropToLayer(net, 'postL2');

        %% query
        x = load(params.input.qlist.path);
        queryImageFilenames = x.query_imgnames_all;

        %serialAllFeats(net, queryDirWithSlash, queryImageFilenames, params.input.feature.dir, 'useGPU', false, 'batchSize', 1);

        nQueries = size(queryImageFilenames,2);
        queryFeatures = struct('queryname', {}, 'features', {});
        for i=1:nQueries
            fprintf('Finding features for query #%d/%d\n\n', i, nQueries)
            queryName = queryImageFilenames{i};
            queryImage = load_query_image_compatible_with_cutouts(fullfile(queryDirWithSlash, queryName), cutoutSize);
            cnn = at_serialAllFeats_convfeat(net, queryImage, 'useGPU', true);
            queryFeatures(i).queryname = queryName;
            queryFeatures(i).features = cnn{6}.x(:);
        end

        %% cutouts
        nCutouts = size(cutoutImageFilenames,2);
        cutoutFeatures = zeros(nCutouts, featureLength, 'single');
        cutoutFeatures = struct('cutoutname', {}, 'features', {});
        for i=1:nCutouts
            fprintf('Finding features for cutout #%d/%d\n\n', i, nCutouts)
            cutoutName = cutoutImageFilenames{i};
            cutoutImage = imread(fullfile(params.dataset.db.cutouts.dir, cutoutName));
            cnn = at_serialAllFeats_convfeat(net, cutoutImage, 'useGPU', true);
            cutoutFeatures(i).cutoutname = cutoutName;
            cutoutFeatures(i).features = cnn{6}.x(:);
        end

        %% save the features
        p = fullfile(params.input.feature.dir, 'computed_features.mat');
        save(p, 'queryFeatures', 'cutoutFeatures', '-v7.3');
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

function [value] = get(structure, field_name, default_value)
    % get Equivalent of Python's dict.get(name, default). 
    if ~isfield(structure, field_name)
        value = default_value;
    else  % Yummy string field name access syntax:
        value = structure.(field_name);
    end
end

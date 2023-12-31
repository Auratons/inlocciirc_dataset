function buildFileLists(varargin)

    addpath(absFnPath('InLocCIIRC_utils', 'params'));
    addpath(absFnPath('yaml'));

    parser = inputParser;
    addOptional(parser, 'config', "use-old-implementation");
    addOptional(parser, 'config_section', "");
    addParameter(parser, 'params', struct());
    parse(parser, varargin{:});

    if parser.Results.config ~= "use-old-implementation"
        % Config fields:
        % input_db_dir
        % input_query_dir
        % output_dir
        % input_db_glob, input_db_depth_glob, input_query_glob
        % output_db_mat_name, output_query_mat_name
        params = ReadYaml(parser.Results.config);
        if parser.Results.config_section ~= ""
            params = params.(parser.Results.config_section);
        end

        %% query
        glob = get(params.file_lists, 'input_query_glob', '*.jpg');
        files = dir(fullfile(params.file_lists.input_query_dir, glob));
        nFiles = size(files, 1);
        query_imgnames_all = cell(1, nFiles);

        for i=1:nFiles
            query_imgnames_all{1, i} = fullfile(files(i).folder, files(i).name);
        end

        [output_dir, ~, ~] = fileparts(params.file_lists.output_query_mat_path);
        if exist(output_dir, 'dir') ~= 7
            mkdir(output_dir);
        end
        save(params.file_lists.output_query_mat_path, 'query_imgnames_all');

        %% database images
        glob = get(params.file_lists, 'input_db_glob', '**/cutout*.jpg');
        files = dir(fullfile(params.file_lists.input_db_dir, glob));
        nFiles = size(files, 1);
        db_imgnames_all = cell(1, nFiles);

        for i=1:nFiles
            db_imgnames_all{1,i} = fullfile(files(i).folder, files(i).name);
        end

        [output_dir, ~, ~] = fileparts(params.file_lists.output_db_mat_path);
        if exist(output_dir, 'dir') ~= 7
            mkdir(output_dir);
        end
        save(params.file_lists.output_db_mat_path, 'db_imgnames_all');

        %% database depth
        glob = get(params.file_lists, 'input_db_depth_glob', '*_depth.npy');
        files = dir(fullfile(params.file_lists.input_db_dir, glob));
        nFiles = size(files, 1);
        db_depthnames_all = cell(1, nFiles);

        for i=1:nFiles
            db_depthnames_all{1,i} = fullfile(files(i).folder, files(i).name);
        end

        [output_dir, ~, ~] = fileparts(params.file_lists.output_db_depth_mat_path);
        if exist(output_dir, 'dir') ~= 7
            mkdir(output_dir);
        end
        save(params.file_lists.output_db_depth_mat_path, 'db_depthnames_all');
    else
        params = setupParams('s10e', true); % TODO: adjust mode

        if exist(params.input.dir, 'dir') ~= 7
            mkdir(params.input.dir);
        end

        %% query
        files = dir(fullfile(params.dataset.query.dir, '*.jpg'));
        nFiles = size(files,1);
        query_imgnames_all = cell(1,nFiles);

        for i=1:nFiles
            query_imgnames_all{1,i} = files(i).name;
        end

        save(params.input.qlist.path, 'query_imgnames_all');

        %% cutouts
        files = dir(fullfile(params.dataset.db.cutouts.dir, '**/cutout*.jpg'));
        nFiles = size(files,1);
        cutout_imgnames_all = cell(1,nFiles);

        for i=1:nFiles
            relativePath = extractAfter(files(i).folder, size(params.dataset.db.cutouts.dir,2)+1);
            cutout_imgnames_all{1,i} = fullfile(relativePath, files(i).name);
        end

        save(params.input.dblist.path, 'cutout_imgnames_all');
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

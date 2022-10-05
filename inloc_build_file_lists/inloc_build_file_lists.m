function inloc_build_file_lists(varargin)
    % Config fields:
    % input_db_dir, input_query_dir
    % input_db_glob, input_query_glob
    % output_db_mat_path, output_query_mat_path
    % Filters requested files and creates list files for later processing.
    % Implementation reworked from the old buildFileLists/buildFileLists.m.

    [filepath, ~, ~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', '..', 'functions', 'inLocCIIRC_utils'));
    inloc_add_abs_fn_path('yaml');

    params = inloc_parse_inputs(varargin{:}).file_lists;

    types = {'db', 'query'};
    for idx = 1:length(types)
        type = types{idx};
        output_mat_path = params.(sprintf('output_%s_mat_path', type));

        if exist(output_mat_path, 'file') ~= 2
            glob = get_with_default(params, sprintf('input_%s_glob', type), '*.jpg');
            files = dir(fullfile(params.(sprintf('input_%s_dir', type)), glob));
            nFiles = size(files, 1);
            filenames = cell(1, nFiles);

            for i=1:nFiles
                filenames{1, i} = fullfile(files(i).folder, files(i).name);
            end

            create_parent_folder(output_mat_path);
            save(output_mat_path, 'filenames');
        else
            fprintf('SKIPPING BUILDING FILE LISTS, output "%s" already exists.\n', output_mat_path);
        end
    end
end

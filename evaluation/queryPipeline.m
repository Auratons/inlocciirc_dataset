queryId = 3;

addpath('../functions/export_fig');
params = struct();
params.dataset.dir = '/Volumes/GoogleDrive/Můj disk/ARTwin/InLocCIIRC_dataset';
params.query.dir = fullfile(params.dataset.dir, 'query');
params.output.dir = fullfile(params.dataset.dir, 'outputs');
params.cutout.dir = fullfile(params.dataset.dir, 'cutouts');
params.densePV.path = fullfile(params.output.dir, 'densePV_top10_shortlist.mat');
params.denseInlier.dir = fullfile(params.output.dir, 'PnP_dense_inlier');
params.synthesized.dir = fullfile(params.output.dir, 'synthesized');
params.evaluation.dir = fullfile(params.dataset.dir, 'evaluation');
params.evaluation.queryPipeline.dir = fullfile(params.evaluation.dir, 'queryPipeline');

queryName = [num2str(queryId), '.jpg'];
queryPath = fullfile(params.query.dir, queryName);
query = imread(queryPath);

load(params.densePV.path, 'ImgList');

fun = @(x) strcmp(ImgList(x).queryname, queryName);
tf = arrayfun(fun, 1:numel(ImgList));
ImgListRecord = ImgList(tf);
cutoutPath = ImgListRecord.topNname{1};
cutoutName = strsplit(cutoutPath, '/');
cutoutName = cutoutName{3};
cutoutName = strsplit(cutoutName, '.');
cutoutName = cutoutName{1};
cutoutPath = fullfile(params.cutout.dir, cutoutPath);
cutout = imread(cutoutPath);

synthPath = fullfile(params.synthesized.dir, queryName, [cutoutName, '.synth.mat']);
load(synthPath, 'RGBpersp', 'errmap');
synth = RGBpersp;

inlierPath = fullfile(params.denseInlier.dir, queryName, [cutoutName, '.pnp_dense_inlier.mat']);
load(inlierPath, 'inls', 'tentatives_2d');
inls_2d = tentatives_2d(:,inls);

if exist(params.evaluation.queryPipeline.dir, 'dir') ~= 7
    mkdir(params.evaluation.queryPipeline.dir);
end

thisQueryPipelineDir = fullfile(params.evaluation.queryPipeline.dir, queryName);

if exist(thisQueryPipelineDir, 'dir') ~= 7
    mkdir(thisQueryPipelineDir);
end

figure;
hold on;
set(gca,'YDir','reverse');
set(gca, 'Visible', 'off');
image(query);
scatter(inls_2d(1,:), inls_2d(2,:), 12, 'green', 'filled');
queryNameNoExt = strsplit(queryName, '.');
queryNameNoExt = queryNameNoExt{1};
queryStepPath = fullfile(thisQueryPipelineDir, ['query_', queryNameNoExt, '.png']);
export_fig(queryStepPath, '-m2');
close;

figure;
hold on;
set(gca,'YDir','reverse');
set(gca, 'Visible', 'off');
image(cutout);
scatter(inls_2d(3,:), inls_2d(4,:), 12, 'green', 'filled');
cutoutStepPath = fullfile(thisQueryPipelineDir, ['chosen_', cutoutName, '.png']);
export_fig(cutoutStepPath, '-m2');
close;

synthStepPath = fullfile(thisQueryPipelineDir, 'synthesized.png');
imwrite(synth, synthStepPath);

figure;
hold on;
set(gca,'YDir','reverse');
set(gca, 'Visible', 'off');
colormap('jet');
errmapStepPath = fullfile(thisQueryPipelineDir, 'errmap.png');
imagesc(errmap);
export_fig(errmapStepPath, '-m2');
close;
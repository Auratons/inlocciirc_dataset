import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
sys.path.insert(1, os.path.join(sys.path[0], '../functions'))
from InLocCIIRC_utils.buildCutoutName.buildCutoutName import buildCutoutName
import matlab.engine
matlabEngine = matlab.engine.start_matlab()

def saveFigure(fig, path, width, height):
    plt.axis('off')
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    img = Image.open(path)
    img = img.resize((width, height), resample=Image.NEAREST)
    img.save(path)

def renderForQuery(queryId, shortlistMode, queryMode, experimentName):
    # shortlistMode is a list that can contain values of {'PV', 'PE'}
    # queryMode is one of {'s10e', 'HoloLens1', 'HoloLens2'}
    inlierColor = '#00ff00'
    inlierMarkerSize = 3
    targetWidth = 600
    cutoutSize = [1600, 1200] # width, height
    targetAspectRatio = cutoutSize[0] / cutoutSize[1]
    targetHeight = np.round(targetWidth / targetAspectRatio).astype(np.int64)
    extension = '.png'

    datasetDir = '/Volumes/GoogleDrive/Můj disk/ARTwin/InLocCIIRC_dataset'

    queryDir = os.path.join(datasetDir, f'query-{queryMode}')
    outputDir = os.path.join(datasetDir, f'outputs-{experimentName}')
    cutoutDir = os.path.join(datasetDir, 'cutouts')

    if shortlistMode == 'PV':
        shortlistPath = os.path.join(outputDir, 'densePV_top10_shortlist.mat')
    elif shortlistMode == 'PE':
        shortlistPath = os.path.join(outputDir, 'densePE_top100_shortlist.mat')
    else:
        raise 'Unsupported shortlistMode!'

    denseInlierDir = os.path.join(outputDir, 'PnP_dense_inlier')
    synthesizedDir = os.path.join(outputDir, 'synthesized')
    evaluationDir = os.path.join(datasetDir, f'evaluation-{experimentName}')
    queryPipelineDir = os.path.join(evaluationDir, 'queryPipeline')

    queryName = str(queryId) + '.jpg'
    queryPath = os.path.join(queryDir, queryName)
    query = plt.imread(queryPath)
    queryWidth = query.shape[1]
    queryHeight = query.shape[0]

    if not os.path.isdir(queryPipelineDir):
        os.mkdir(queryPipelineDir)
    
    shortlistModeDir = os.path.join(queryPipelineDir, shortlistMode)
    if not os.path.isdir(shortlistModeDir):
        os.mkdir(shortlistModeDir)

    thisParentQueryDir = os.path.join(shortlistModeDir, queryName)
    if not os.path.isdir(thisParentQueryDir):
        os.mkdir(thisParentQueryDir)

    ImgList = sio.loadmat(shortlistPath, squeeze_me=True)['ImgList']
    ImgListRecord = next((x for x in ImgList if x['queryname'] == queryName), None)
    topNname = ImgListRecord['topNname']
    if topNname.ndim == 1:
        cutoutNames = [topNname[0]]
    else:
        cutoutNames = topNname[:,0]

    if shortlistMode == 'PV':
        dbnamesId = ImgListRecord['dbnamesId'][0]
    elif shortlistMode == 'PE':
        dbnamesId = 1

    synthPath = os.path.join(synthesizedDir, queryName, f'{dbnamesId}.synth.mat')
    synthData = sio.loadmat(synthPath, squeeze_me=True)
    inlierPath = os.path.join(denseInlierDir, queryName, f'{dbnamesId}.pnp_dense_inlier.mat')
    inlierData = sio.loadmat(inlierPath)
    segmentLength = len(cutoutNames)
    for i in range(segmentLength):
        thisQueryName = str(queryId - segmentLength + i + 1) + '.jpg'
        print(f'Processing query {thisQueryName}, as part of the segment')
        if segmentLength == 1:
            synth = synthData['RGBpersps']
            errmap = synthData['errmaps']
        else:
            synth = synthData['RGBpersps'][i]
            errmap = synthData['errmaps'][i]
        inls = inlierData['allInls'][0,i]
        tentatives_2d = inlierData['allTentatives2D'][0,i]
        cutoutName = cutoutNames[i]
        inls = np.reshape(inls, (inls.shape[1],)).astype(np.bool)
        inls_2d = tentatives_2d[:,inls] - 1 # MATLAB is 1-based
        thisQueryPath = os.path.join(queryDir, thisQueryName)
        thisQuery = matlabEngine.load_query_image_compatible_with_cutouts(thisQueryPath, matlab.double(cutoutSize), nargout=1)
        thisQuery = np.asarray(thisQuery)

        cutout = plt.imread(os.path.join(cutoutDir, cutoutName))

        thisQueryPipelineDir = os.path.join(thisParentQueryDir, thisQueryName)
        if not os.path.isdir(thisQueryPipelineDir):
            os.mkdir(thisQueryPipelineDir)

        fig = plt.figure()
        plt.imshow(thisQuery)
        plt.plot(inls_2d[0,:], inls_2d[1,:], '.', markersize=inlierMarkerSize, color=inlierColor)
        thisQueryNameNoExt = thisQueryName.split('.')[0]
        queryStepPath = os.path.join(thisQueryPipelineDir, 'query_' + thisQueryNameNoExt + extension)
        saveFigure(fig, queryStepPath, targetWidth, targetHeight)
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(cutout)
        plt.plot(inls_2d[2,:], inls_2d[3,:], '.', markersize=inlierMarkerSize, color=inlierColor)
        cutoutStepPath = os.path.join(thisQueryPipelineDir, 'chosen_' + buildCutoutName(cutoutName, extension))
        saveFigure(fig, cutoutStepPath, targetWidth, targetHeight)
        plt.close(fig)

        synthStepPath = os.path.join(thisQueryPipelineDir, 'synthesized' + '.PV' + extension)
        synth = Image.fromarray(synth)
        synth = synth.resize((queryWidth, queryHeight), resample=Image.NEAREST)
        synth = np.asarray(synth)
        plt.imsave(synthStepPath, synth)

        # NOTE: the errmap typically does not have the same aspect ratio, so it will be stretched
        errmapStepPath = os.path.join(thisQueryPipelineDir, 'errmap' + extension)
        errmap = Image.fromarray(errmap)
        errmap = errmap.resize((targetWidth, targetHeight), resample=Image.NEAREST)
        errmap = np.asarray(errmap)
        plt.imsave(errmapStepPath, errmap, cmap='jet')

matlabEngine.addpath(r'functions/InLocCIIRC_utils/at_netvlad_function',nargout=0)
queryMode = 'HoloLens1'
experimentName = 'HL1-v4-k5'
shortlistModes = ['PV']
queryIds = [1,127,200,250,100,300,165,55,330,223]
for shortlistMode in shortlistModes:
    for queryId in queryIds:
        print(f'[{shortlistMode}] Processing query {queryId}.jpg segment')
        renderForQuery(queryId, shortlistMode, queryMode, experimentName)
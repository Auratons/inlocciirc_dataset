# InLoc dataset transformations

This is a fork of https://github.com/lucivpav/InLocCIIRC_dataset

For the purpose of https://github.com/Auratons/master_thesis,
only `buildCutouts/build_cutouts_*.py` scipts are used, targeting
datasets by name with their specific internal format.

## Old README contents

This repository contains tools for building the InLocCIIRC dataset.
The dataset is constructed incrementally for each Space.
The usual steps are as follows:

1. Obtain the sweepData.json file
2. Obtain panoramas
3. Obtain MatterPak and normalize it
4. Rotate panoramas
5. Build cutouts
6. Build point cloud file
7. Build query poses
8. Build file lists
9. Build scores
10. Plot the dataset including retrieved poses
11. Plot query pipeline
12. Prepare to plot distance threshold vs accuracy

### Obtain the sweepData.json file
1. Create a key.js file based on the keyTemplate.js file
2. Run a web server in the getSweepData folder
3. Open the getSweepData.html in your browser **as a localhost** address - e.g. http://127.0.0.1:8887/getSweepData.html
4. Open the console, the sweepData.json is being printed there

### Obtain panoramas
* Manually download a panorama for every sweep
* Name the panoramas according to their number as taken by the Capture iPadOS app
* Make sure the circle around the mouse pointer is not present in the panorama
* Name the panoramas as *number*.pano in matterport.com
* Download the panoramas as *number.jpg*

### Obtain MatterPak and normalize it
1. Buy the MatterPak
2. Download it, it contains cloud.xyz, .obj files
3. Rotate them along the x axis (psi angle) by -90.0 degrees; recommended tool: CloudCompare
4. Save them accordingly into the models directory, use .ply extension for the point cloud, .obj extension for the mesh

### Rotate panoramas
1. Open the rotatePanoramas folder in Matlab
2. Set up the appropriate Space name in setupParams2.m
3. Adjust and run buildSweepDataMatFile.m
4. Adjust and run rotatePanoramas.m
5. For panoramas that failed to rotate properly, try changing the *goodness* in sweepData
6. Try increasing the point size of the point cloud projection
7. If the proper rotation still cannot be found, use manuallyRotatePanorama.m file

### Build cutouts
1. Adjust the Space name and the panoIds array
2. It is necessary that the display is turned on, otherwise you get an error from pyrender

### Build point cloud file
1. Adjust the Space name

### Build query poses
1. Choose the desired mode in transformPoses.m: setupParams(mode)

### Build file lists
1. Note the comment on the third line
2. Change the mode accordingly

### Build scores
1. Set up appropriate mode in buildFeatures.m
2. Execute buildFeatures.m on a machine with GPU.
3. Execute buildScores on a machine with ~1 GB of RAM

### Plot the dataset including retrieved poses 
1. Make sure the demo has finished and now we have retrievedPoses directory in evaluation directory
2. It is recommended to erase evaluation/temporary directory
3. Run evaluation/spaceTopViews.py and check that the output images are looking good

### Plot query pipeline
1. Run evaluation/queryPipeline.m for queries of interest

### Prepare to plot distance threshold vs accuracy
1. Execute evaluation/distThreshVsAccuracy.py

### TODO
* empty TODO list

### Sequences
If you have query sequences, you need to generate raw poses. TODO: the next steps are outdated!

1. Navigate to buildRawPoses.m
2. Guess a synchronization constant
3. Set up parameters in *functions/InLocCIIRC_utils/params* folder
4. Set generateMiniSequence to true, until you find the right synchronization constant
5. Set generateMiniSequence to false, adjust the params until projections match queries
6. Next, when a not-great-not-terrible generic params are found, use the next section to build upon them and find even better params
7. Once the parameters are good enough, scroll down and execute the code snippet that generates rawPoses.csv

### WARNINGS
* Do not overwrite functions/matconvnet when syncing local repo with remote repo. It contains already built package for the remote. However, if hit accidentally happens, prebuilt binary can be found at boruvka:/datagrid/personal/lucivpav/matconvnet
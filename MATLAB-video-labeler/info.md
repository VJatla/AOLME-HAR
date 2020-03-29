This directory contains scripts to process ground truth produced using
MATLAB 2018b Video Labeler application.

## Files:
	1. `+vision/+labeler/fixed_bbox.m`,
		Fixed bounding box class for automation engine.
		It worked with MATLAB 2018b
	2. `mat_to_csv.m`, 
		Extracts ground truth from groundTruth object file to CSV file.
	3. `bboxes_to_nparrays.py`,
		Extracting activity instances as trimmed numpy arrays.

## Desription:
### Fixed bounding box (Automation Algorithm)
The following automation algorithm for *Video Labeler* allows 
fixed bounding boxes. Basically location and size of the
bouding box does not change.

To use this algorithm make sure to place it in `+vision/+labeler` directory.
Also add this path to MATLAB search paths.

**Example**
My file path,
```bash
C:\Users\vj\Dropbox\Marios_Shared\AOLME-HAR-root\software\AOLME-HAR\ground-truth\MATLAB-video-labeler\+vision\+labeler\fixed_bbox.m
```
To make this algorithm work I have to add
```bash
C:\Users\vj\Dropbox\Marios_Shared\AOLME-HAR-root\software\AOLME-HAR\ground-truth\MATLAB-video-labeler
```
to my path, using `Set Path`(near Preferences) button provided by MATLAB.

### mat to csv
The following script reads bounding boxes tracked using *point tracker* or *fixed_bbox*
automation algorithm to csv file. This script addresses the issue of varying size and
position of bounding box (when using point tracker) by fixing the position and size to
initialization.
The output csv has same name as the `mat` file with following columns
```bash
???
```


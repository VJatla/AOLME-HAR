"""
The following script reads bounding boxes from all CSV files under a root
directory and extracts activities as trimmed videos. All the trimmed
videos are stored under corresponding directories.

The CSV files having human activities marked should have following columns,

| name | activity | person | W | H | FPS | w0 | h0 | f0 | w | h | f |

    + name     = name of the video file.
    + activity = Activity performed in the bounding box.
    + person   = Person identity who is performing that action.
    + W        = Width of the video.
    + H        = Height of the video.
    + FPS      = Frame rate of the video.
    + w0       = Top left pixel location of bounding box along width.
    + h0       = Top left pixel location of bounding box along height.
    + f0       = Initial frame number of bounding box (POC, starts from 0).
    + w        = width of bounding box.
    + h        = height of bounding box.
    + f        = Final frame number of bounding box (POC, starts form 0).,

Note: Pixel indexing starts at top left. In other words, top left corner
      of the frame has (0,0) pixel indexing.
"""

import pdb
import vlearn


# Initializations
rdir          = "../../../../../writing-nowriting-GT/C1L1P-C/20170225/"
odir          = "/home/vj/Software/mmaction/data/aolme_wnw_C1L1PC/videos/"
gt_csv_name   = "gTruth-wnw.csv"

# Extract activities in the same directory and copy to output directory
dirs_with_gtcsv = vlearn.file_dir_ops.list_unique_dirs(rdir, [gt_csv_name])
for cur_dir in dirs_with_gtcsv:
    acubes    = vlearn.ActCubes(cur_dir, gt_csv_name)
    acubes.extract_activity_cubes(out_dir=odir)

"""
The following script reads bounding boxes from csv files and
summarizes their properties. This is mainly used to summarize Human Activities.
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
import matplotlib.pyplot as plt

rdir     = "C:\\Users\\vj\\Dropbox\\writing-nowriting-GT\\C1L1P-C"
acubes   = vlearn.ActCubes(rdir, "gTruth-wnw.csv")
acubes.plot_properties("nowriting")
acubes.plot_properties("writing")
plt.show()

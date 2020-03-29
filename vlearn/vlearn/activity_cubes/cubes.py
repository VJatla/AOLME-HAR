import os
import cv2
import scipy
import math
import numpy as np
import pandas as pd
import pdb
from ..file_dir_ops import list_files
from ..vid_reader import VReader
import matplotlib.pyplot as plt


class ActCubes(VReader):
    def __init__(self, rdir, gt_csv_name):

        """
        Description:
            Initializes a pandas data frame with activity cubes. These are
            read from csv files present under root directory 
            (including sub-directories)
        Args:
            rdir (str): 
                Root directory containing csv files which contain activity cubes
            csv_name (str):
                Name of csv files that has ground truth. The assuption is that
                the ground truth have same name.
        Example:
            ```
            import vlearn
            ac = vlearn.ActCubes("./training")
            ```
        """
        self.rdir = rdir
        """ Root directory having ground truth csv file and videos """

        self.cubes = pd.DataFrame()
        """ Data frame having activity cube information """

        all_csv_files = list_files(rdir, [gt_csv_name])
        for idx, ccsv in enumerate(all_csv_files):
            if idx == 0:
                self.cubes = pd.read_csv(ccsv)
            else:
                tmp_cubes = pd.read_csv(ccsv)
                tmp_cubes = [tmp_cubes, self.cubes]
                self.cubes = pd.concat(tmp_cubes)
                

    def plot_properties(self, activity):
        """
        Description:
            Creates histogram of  width, height and number of frames.
            These histograms are returned as list of axis handles. To
            view them please use `plt.imshow()`
        
        Args:
            activity (str):
                Activity under consideration.
        """
        cur_act_cubes = self.cubes[self.cubes["activity"] == activity] 
        warr          = np.array(cur_act_cubes["w"])
        harr          = np.array(cur_act_cubes["h"])
        farr          = np.array(cur_act_cubes["f"])
        fpsarr        = np.rint(np.array(cur_act_cubes["FPS"]))
        tarr          = np.divide(farr,fpsarr)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        wax = self.__plot_histogram(
            warr, "Width(pixels)", "Count", "Width histogram for " + activity
        )
        textstr = "Min " + str(warr.min())+ "\n" +\
            "Max " + str(warr.max())+ "\n" +\
            "Avg " + str(np.rint(warr.sum()/len(warr)))
        wax.text(0.75, 0.95, textstr, transform=wax.transAxes, fontsize=28,
                 
         verticalalignment='top', bbox=props)
        
        hax = self.__plot_histogram(
            harr, "Height(pixels)", "Count", "Height histogram for " + activity
        )
        textstr = "Min " + str(harr.min())+ "\n" +\
            "Max " + str(harr.max())+ "\n" +\
            "Avg " + str(np.rint(harr.sum()/len(harr)))
        hax.text(0.75, 0.95, textstr, transform=hax.transAxes, fontsize=28,
                 verticalalignment='top', bbox=props)
        
        tax = self.__plot_histogram(
            tarr, "Time in seconds", "Count", "Play back time  for " + activity
        )
        textstr = "Tot " + str(np.round(tarr.sum()/3600, 2)) + " hrs\n" +\
            "Min " + str(round(tarr.min(), 2))+ " sec\n" +\
            "Max " + str(round(tarr.max(), 2))+ " sec\n" +\
            "Avg " + str(np.rint(tarr.sum()/len(tarr))) + " sec\n"
        tax.text(0.75, 0.95, textstr, transform=tax.transAxes, fontsize=28,
                 verticalalignment='top', bbox=props)
        
    def __plot_histogram(self, arr, xlab, ylab, title):
        """
        Description:
            Creates histogram using a numpy array.
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        kwargs = dict(histtype="stepfilled", alpha=0.8, bins=20)
        ax.hist(arr, **kwargs)
        ax.set_xlabel(xlab, fontsize=30)
        ax.set_ylabel(ylab + " ("+str(len(arr))+" samples)", fontsize=30)
        ax.tick_params(axis="both", which="major", labelsize=25)
        ax.tick_params(axis="both", which="minor", labelsize=25)
        ax.set_title(title, fontsize=35)
        return ax

    def extract_activity_cubes(self, dur=3, out_dir=""):
        """
        Trims activity cubes and stores them as videos in the same
        directory as CSV file. Once extracted the files are copied
        to `out_dir` if the argument is not empty.
        Args:
            dur (int): Duration of each activity cube. By default it is set to
                3 seconds.
            out_dir (str): Output location of trimmed videos. If not passed videos
                are generated at the same location as video (csv file).
            np_array (boolen): When true generates a numpy array for all the trimmed
                videos.
        """

        
        for idx, row in self.cubes.iterrows():
            # Read video for current activity cube
            vid_name = row["name"]
            vpath = self.rdir + "/" + vid_name
            super(ActCubes, self).__init__(vpath)
            # Number of trims. A 10 second video is trimmed 3 times.
            frames_per_trim = dur*math.ceil(row["FPS"])
            ntrims          = math.floor(row["f"] / frames_per_trim)
            
            for ctrim in range(0, ntrims):
                poc           = row["f0"] + frames_per_trim * ctrim
                poce          = poc + frames_per_trim
                trim_name     = row["person"] + "_" + self.vname + "_" +\
                    str(poc) + "_" + str(poce - 1) + ".avi"
                trim_path     = self.rdir + "/" + row["activity"]
                out_path      = out_dir   + "/" + row["activity"] + "/"
                trim_fpath    = trim_path + "/" + trim_name

                # Create directories if they do not exist
                if not(os.path.isdir(trim_path)):
                    os.mkdir(trim_path)
                if not(os.path.isdir(out_path)):
                    os.mkdir(out_path)

                # Using  PNG format to avoid compression artifacts
                print(trim_fpath)
                vwr = cv2.VideoWriter(trim_fpath,
                                      cv2.VideoWriter_fourcc("p","n","g"," "),
                                      row["FPS"],
                                      (100,100)
                )
                self.vro.set(cv2.CAP_PROP_POS_FRAMES, poc)
                while self.vro.isOpened() and poc < poce:
                    ret, frm = self.vro.read()
                    trm_frm  = frm[ int(row["h0"]) : int(row["h0"] + row["h"]),
                                    int(row["w0"]) : int(row["w0"] + row["w"])]
                    trm_frm  = cv2.resize(trm_frm, (100, 100))
                    vwr.write(trm_frm)
                    poc      = poc + 1
                    
                vwr.release()
                
                # Copying files to output directory
                cp_cmd = "cp " + trim_fpath + " " + out_path
                os.system(cp_cmd)
            
            self.vro.release()

    def extract_activity_cubes_as_np_arrays(self, frames, out_dir):
        """
        Trims activity cubes and stores them as videos. For
        writing and nowriting wenjing suggested to use 30
        frames or 1 second.
        Args:
            frames (int): Number of frames in each cube.
            out_dir (str): Output location of trimmed videos. If not passed videos
                are generated at the same location as video (csv file).
        """
        num_gt_cubes = len(self.cubes)
        gray_list = []
        for idx, row in self.cubes.iterrows():
            # Read video for current activity cube
            matlab_gt_name = row["name"]
            vid_name = matlab_gt_name.replace("_vj_gTruth", "")  # ??? Hard coded
            vpath = self.rdir + "/" + vid_name
            super(ActCubes, self).__init__(vpath)
            # Trim videos every n frames
            ntrims = math.floor(row["f"] / frames)
            for ctrim in range(0, ntrims):
                poc = row["f0"] + frames * ctrim
                poce = poc + frames
                trim_name = vid_name + "_" + str(poc) + "_" + str(poce - 1)
                trim_path = self.rdir + "/" + vid_name + "_trimmed_nparr"
                if not (os.path.isdir(out_dir + "/nparr_" + row["activity"])):
                    os.mkdir(out_dir + "/nparr_" + row["activity"])
                    trim_fpath = (
                        out_dir
                        + "/nparr_"
                        + row["activity"]
                        + "/"
                        + row["person"]
                        + "_"
                        + trim_name
                        + ".npy"
                    )
                    print("Saving ", trim_fpath)
                    self.vro.set(cv2.CAP_PROP_POS_FRAMES, poc)
                    k = 0
                    trm_arr = np.ndarray((frames, 100, 100, 3))
                while self.vro.isOpened() and poc < poce:
                    ret, frm = self.vro.read()
                    trm_frm = frm[
                        int(row["h0"]) : int(row["h0"] + row["h"]),
                        int(row["w0"]) : int(row["w0"] + row["w"]),
                    ]
                    trm_frm = cv2.resize(trm_frm, (100, 100))
                    trm_arr[k, :, :, :] = trm_frm
                    poc = poc + 1
                    k = k + 1
                    np.save(trim_fpath, trm_arr.astype("uint8"))
                    self.vro.release()

import os
import cv2
import pdb
import sys


class VReader:
    def __init__(self, vpath):
        """
        This class is primarily designed to act as parent class to other
        classes which needs to read video.
        """

        self.vname = os.path.basename(vpath).split(".")[0]
        """ Name of video  """

        self.vpath = os.path.dirname(vpath)
        """ Absolute path of the video """

        self.vro = cv2.VideoCapture(vpath)
        """ OpenCV VideoReader instance. This can be used
            to read video and its properties."""

        if not (self.vro.isOpened()):
            print("Error opening video ")
            print("\t ", vpath)
            sys.exit(1)

        self.vfps = round(self.vro.get(cv2.CAP_PROP_FPS))
        """ Frame per second """

        self.vht = int(self.vro.get(cv2.CAP_PROP_FRAME_HEIGHT))
        """ Frame height """

        self.vwd = int(self.vro.get(cv2.CAP_PROP_FRAME_WIDTH))
        """ Frame Width """

        self.nfrms = int(self.vro.get(cv2.CAP_PROP_FRAME_COUNT))
        """ Number of frames in video """

        self.ply_time = self.nfrms / self.vfps
        """ Play back time """

    def extract_frames(self, pocs=0, poce=-1, skip=-1, out_dir=""):
        """
        Extracts a video as `png` images. The frames are in a directory created
        with same name as video. This directory is created in the same location
        as video.
        Args:
            pocs (int): Starting frame number. By default it is 0.
            poce (int): Ending frame number. By default it is -1.
                -1 implies that frames are extracted till the end.
            skip (int): Number of frames to skip. By default it is equal
                FPS of video. That means we get one frame for every 1
                second.
        """
        if poce == -1:
            poce = self.nfrms - 1
        if skip == -1:
            skip = self.vfps

        # setting video to starting POC (pocs)
        poc = pocs
        self.vro.set(cv2.CAP_PROP_POS_FRAMES, poc)

        # Loop over the video
        while self.vro.isOpened() and poc <= poce:
            ret, frm = self.vro.read()
            img_name = self.vname + "_" + str(poc) + ".png"

            if out_dir == "":

                img_path = self.vpath + "/" + self.vname
                if not (os.path.isdir(img_path)):
                    os.mkdir(img_path)
                img_fpath = img_path + "/" + img_name

            else:

                img_dir_path = out_dir + "/" + self.vname
                if not (os.path.isdir(img_dir_path)):
                    os.makedirs(img_dir_path)
                img_fpath = img_dir_path + "/" + img_name

            print("Writing \t", img_fpath)
            cv2.imwrite(img_fpath, frm)

            poc += skip
            if skip > 1:
                self.vro.set(cv2.CAP_PROP_POS_FRAMES, poc)

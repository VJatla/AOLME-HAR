import pdb
import sys
import cv2
import numpy as np
from ..vid_reader import VReader

ERROR_CODE = -2


class OptFlw(VReader):
    def __init__(self, vpath, method="Farneback"):
        """
        Initializes an instance that uses optical flow to determine active
        regions in a video.
        
        Args:
            vpath (str): Video file path
            method (str,optional): Optical flow method to use. It takes 
                following options, 1. Farneback (default), 2. ????
        """
        super(OptFlw, self).__init__(vpath)
        self._method = "Farneback"

    def get_nmag_map_vid(self, pocs, poce):
        """
        Returns a numpy array with same resolution as video.
        Each element in this array is normalized optical flow
        magnitude.

        Args:
            pocs (int): Starting poc
            poce (int): Ending poc. Value of -1 implies we use all frames.
        """
        # poce to ending POC in case -1 is given
        if poce == -1:
            poce = self.nfrms - 1

        # setting video to starting POC (pocs)
        poc = pocs
        self.vro.set(cv2.CAP_PROP_POS_FRAMES, poc)

        # Reading first frame
        ret, frm0 = self.vro.read()
        poc += 1

        # Loop over all the remaining frames till poce
        vid_mag_map = np.zeros((self.vht, self.vwd))
        while self.vro.isOpened() and ret and poc <= poce:
            print(poc)
            ret, frm1 = self.vro.read()
            poc += 1

            # Calculate magnitude map for current frames
            frm_mag_map = self.get_mag_map_frms(frm0, frm1)

            # Collect in video magnitude map
            vid_mag_map = vid_mag_map + frm_mag_map

        # Normalize video magnitude map to 0 and 1
        vid_mag_map = cv2.normalize(vid_mag_map, None, 0, 1, cv2.NORM_MINMAX)

        return vid_mag_map

    def get_mag_map_frms(self, frm0, frm1):
        """
        Computes optical flow magnitude map for
        between two frames.
        Args:
            frm0 (ndarray): Video frame 0
            frm1 (ndarray): Video frame 1
        Note:
           frm1 plays later than frm0. That means poc(frm0) < poc(frm1)
        """
        # Convert to gray scale
        frm0_gray = cv2.cvtColor(frm0, cv2.COLOR_BGR2GRAY)
        frm1_gray = cv2.cvtColor(frm1, cv2.COLOR_BGR2GRAY)

        # Choose optical flow method to use
        if self._method == "Farneback":

            # Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                frm0_gray, frm1_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, ang_deg = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], True)

        else:
            print("Invalid optical flow method selected ")
            print(self.method)
            sys.exit(ERROR_CODE)

        return mag

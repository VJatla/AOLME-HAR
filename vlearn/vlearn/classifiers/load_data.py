import os
import cv2
import sys
import pdb
import numpy as np
import pandas as pd
from ..file_dir_ops import list_files


class DataLoader:
    """
    Loads data as numpy arrays.
    """

    def __init__(self, params):
        """
        Loads data.

        Args:
            floc (str):
                Path to a txt file having training videos and its
                corresponding label.
            ch_typ (str): 
                Type of video channel used for training. It can 
                take following values, {gray, flow}                
        """
        self._tr_file_loc = params["train_videos"][0]
        self._ts_file_loc = params["test_videos"][0]
        self._vl_file_loc = params["val_videos"][0]
        self._data_dir = params["data_dir"][0]
        self._channel_type = params["channel_type"][0]


    def load_as_np_arrays(self, split_type):
        """
        Returns a 4D numpy array and corresponding label array.

        Args:
            split_type (str): Data split to process. It takes following
                values, {training, testing}
        """
        if split_type == "training":
            file_loc = self._tr_file_loc
        elif split_type == "testing":
            file_loc = self._ts_file_loc
        elif split_type == "validation":
            file_loc = self._vl_file_loc
        else:
            print("Does not support loading ", split_type)
            sys.exit()

        fdir = os.path.dirname(file_loc)
        fname = os.path.basename(file_loc).split(".")[0]
        X_fpath = fdir + "/" + self._channel_type + "_X_" + fname + ".npy"
        y_fpath = fdir + "/" + self._channel_type + "_y_" + fname + ".npy"

        if os.path.exists(X_fpath) and os.path.exists(y_fpath):

            print("Loading existing numpy arrays")
            print("\t", X_fpath)
            print("\t", y_fpath)

            X = np.load(X_fpath)
            y = np.load(y_fpath)
        else:

            print("Generating and savign numpy arrays")
            print("\t", X_fpath)
            print("\t", y_fpath)

            if self._channel_type == "gray":
                X, y = self._gen_gray_np_array(file_loc)
                np.save(X_fpath, X)
                np.save(y_fpath, y)
            elif self._channel_type == "flow":
                X, y = self._gen_flow_np_array(file_loc)
                np.save(X_fpath, X)
                np.save(y_fpath, y)
            else:
                print("ERROR: Unrecognized channel ", self._channel_type)

        return X, y

    def _gen_gray_np_array(self,file_loc):
        """
        Generates gray numpy array from videos present in the text
        file.
        """
        # Reading training list as pandas dataframe
        vid_lst_df = pd.read_csv(file_loc, sep=" ", header=None)

        # Getting activity cube dimensions from first video in video list dataframe
        vid_name = os.path.basename(vid_lst_df[0][0])
        vid_loc = list_files(self._data_dir, [vid_name])
        num_frms, num_cols, num_rows = self._get_vid_dims(vid_loc[0])
        num_channels = 1

        # Creating X and y np arrays
        X = np.ndarray(
            (len(vid_lst_df), num_frms, num_rows, num_cols, num_channels), dtype="uint8"
        )
        y = np.array([-1] * len(vid_lst_df))

        # Loop over videos in the video list dataframe
        for i, row in vid_lst_df.iterrows():
            vid_name = os.path.basename(row[0])
            vid_loc = list_files(self._data_dir, [vid_name])

            # If more than 1 video is found matching throw error
            if len(vid_loc) > 1:
                print("ERROR: More than 1 video location found")
                print("\t", vid_loc)
                sys.exit()

            # Load current video and read first frame
            X[i,:,:,:,0] = self._get_gray(vid_loc[0])

            # Creating y array having labels read from video list dataframe
            y[i] = row[1]

        # return X and y
        return X, y


    def _gen_flow_np_array(self,file_loc):
        """
        Generates optical flow numpy array from videos present in the text
        file.
        """
        print("Doesnot work with numpy arrays, should be fixed???")
        sys.exit()
        # Reading training list as pandas dataframe
        vid_lst_df = pd.read_csv(file_loc, sep=" ", header=None)

        # Getting video properties from first video in video list dataframe
        vid_name = os.path.basename(vid_lst_df[0][0])
        vid_loc = list_files(self._data_dir, [vid_name])
        vid = cv2.VideoCapture(vid_loc[0])
        num_frms = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frm = vid.read()
        num_rows, num_cols, _ = frm.shape
        num_channels = 2

        # Creating X and y np arrays
        X = np.ndarray(
            (len(vid_lst_df), num_frms-1, num_rows, num_cols, num_channels), dtype="uint8"
        )
        y = np.array([-1] * len(vid_lst_df))

        # Initializing tvl1 instance
        tvl1 = cv2.DualTVL1OpticalFlow_create()
        
        # Loop over videos in the video list dataframe
        for i, row in vid_lst_df.iterrows():
            vid_name = os.path.basename(row[0])
            vid_loc = list_files(self._data_dir, [vid_name])

            # If more than 1 video is found matching throw error
            if len(vid_loc) > 1:
                print("ERROR: More than 1 video location found")
                print("\t", vid_loc)
                sys.exit()

            # Load current video and read first frame
            vid = cv2.VideoCapture(vid_loc[0])
            ret, frm0 = vid.read()
            frm_num   = 0
            gray_frm0 = cv2.cvtColor(frm0,cv2.COLOR_BGR2GRAY)
            ret, frm1 = vid.read()
            frm_num   += 1

            # Loop over other frames
            while vid.isOpened() and ret:
                gray_frm1  = cv2.cvtColor(frm1,cv2.COLOR_BGR2GRAY)
                # flow = cv2.calcOpticalFlowFarneback(gray_frm0, gray_frm1,None,
                #                                     0.5, 3, 15, 3, 5, 1.2, 0)
                flow = tvl1.calc(gray_frm0,gray_frm1,None)
                X[i,frm_num-1,:,:,:] = flow
                #New frame
                gray_frm0  = gray_frm1.copy()
                ret, frm1  = vid.read()
                frm_num  += 1
            # Creating y array having labels read from video list dataframe
            y[i] = row[1]

        # return X and y
        return X, y

    def _get_vid_dims(self, vpath):
        """
        Returns video dimenstions.
        Args:
            vpath(str) : Path to video having activity
        """
        if ".npy" in vpath:
            npy_arr = np.load(vpath)
            num_frms, num_rows, num_cols, _ = npy_arr.shape
        elif ".avi" in vpath:
            vid = cv2.VideoCapture(vpath)
            num_frms = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, frm = vid.read()
            num_rows, num_cols, _ = frm.shape
        else:
            print("The current file type is not supported")
            print(vpath)
            sys.exit()

        return num_frms, num_cols, num_rows


    def _get_gray(self, vpath):
        """
        Returns a video as gray np arrya (uint8)
        Args:
            vpath(str) : Path to video having activity
        """
        if ".npy" in vpath:
            
            npy_arr = np.load(vpath)
            nfrms, nrows, ncols, nch = npy_arr.shape
            gray_vid = np.ndarray((nfrms, nrows, ncols))
            for i in range(0,nfrms):
                gray_vid[i,...] = cv2.cvtColor(npy_arr[i,...],cv2.COLOR_BGR2GRAY)
                
        elif ".avi" in vpath:
            
            vid = cv2.VideoCapture(vpath)
            nfrms = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, frm = vid.read()
            nrows, ncols, nch = frm.shape
            gray_vid = np.ndarray((nfrms, nrows, ncols))
            # Loop over other frames
            i = 0
            while vid.isOpened() and ret:
                gray_frm = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                gray_vid[i,...] = gray_frm  # 0 channel
                i += 1
                ret, frm = vid.read()
                
        else:
            print("The current file type is not supported")
            print(vpath)
            sys.exit()

        return gray_vid.astype("uint8")

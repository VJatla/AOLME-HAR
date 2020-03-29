import os
import cv2
import pdb
import glob
import numpy as np
from ..file_dir_ops import list_files

class ImgAug:
    """
    """

    def __init__(self, dir_loc, img_ext=".png", list_methods = ["CLAHE"] ):
        """
        The following class provides methods to augment image data.
        
        Args:
            dir_loc (str): String having locaiton of directory having images.
                 Please give the last level of directory tree. Make sure that
                 there are no sub directories having images.
            img_ext (str): Image extenstion as stirng. Default is ".png"
            list_methods (str, list): Image augmentation methods
        """
        # Directory name
        dir_name = os.path.basename(dir_loc)
        dir_path = os.path.join(dir_loc,"..")

        # Get list of images in current directory
        img_list = glob.glob(dir_loc+"/*.png")

        # Loop over each image
        for img_path in img_list:
            img      = cv2.imread(img_path)
            

            if ("CLAHE" in list_methods):
                img_lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                img_l       = img_lab[...,0]
                clahe       = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l_clahe     = clahe.apply(img_l)
                img_clahe   = cv2.merge((l_clahe,
                                         img_lab[...,1],
                                         img_lab[...,2]))
                img_clahe   = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)
                # Create director if not existing
                clahe_path = dir_path + "/" + dir_name + "_clahe"
                if (not(os.path.isdir(clahe_path))):
                    os.mkdir(clahe_path)
                # write image
                clahe_img_path = clahe_path + "/" + os.path.basename(img_path)
                cv2.imwrite(clahe_img_path, img_clahe)

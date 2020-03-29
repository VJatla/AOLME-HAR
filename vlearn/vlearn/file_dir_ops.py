import os
import pdb
import sys
import pandas as pd


def list_files(root_dir, kw_lst):
    """
    DESCRIPTION:
        This function lists full paths of files containing keywords the
        user needs. These keywords are provided as sencond argument.

    INPUT:
        root_dir: directory path
        kw_lst  : list of strings containing keywords

    Example:
        import vlearn

        # creates a list having csv file paths with gTruth in their name
        csv_lst = vlearn.file_dir.list_files("~/Documents",["gTruth",".csv"])
    """

    # Check if directory is valid
    if not (os.path.exists(root_dir)):
        print("The following path does not exist")
        print(root_dir)
        sys.exit(1)

    # Loop through each file
    files = []
    for r, d, f in os.walk(root_dir):
        for file in f:
            # Check if current file contains all of the key words
            is_valid_file = all(kw in file for kw in kw_lst)
            if is_valid_file:
                files.append(os.path.join(r, file))

    # return
    return files


def copy_files(lst, dst):
    """

    DESCRIPTION:
        Copies all files from the list to destination folder

    INPUT:
        lst = list of files
        dst = destination path
    """
    # csv file loop
    for f in lst:
        print("Copying ", f)
        cmd = "cp " + f + " " + dst
        os.system(cmd)

def list_unique_dirs(rdir, kw_lst):
    """
    DESCRIPTION:
       Creates a list of directoires having specific files.

    INPUT:
        root_dir: directory path
        kw_lst  : list of strings containing keywords
    """
    file_paths = list_files(rdir, kw_lst)
    paths      = [os.path.dirname(os.path.abspath(x))
                  for x in file_paths]
    uniq_paths = list(set(paths))

    return uniq_paths

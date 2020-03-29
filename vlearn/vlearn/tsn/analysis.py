"""
The following source contains methods that help in comparing and analyzing 
TSN runs.

Ref:
    https://ieeexplore.ieee.org/document/8454294
"""


import pdb
import time
import json
import math
import torch
import numpy as np
import pandas as pd
from ..file_dir_ops import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# MMACTION/MMCV libraries
import mmcv
from mmcv.runner import load_checkpoint
from mmaction.models import build_recognizer



class Analyze:
    """
    Class having methods and techiques that help in analyzing and comparing
    TSN runs.
    """


    def __init__(self, dirs_, cfg_file):
        """
        Initializes TSN analysis instance with list of directories having
        checkpoint files and logs. A comparision ,
            1. Log file in JSON format.
            2. Checkpoint files stored in `pth` format.

        args:
            dris_: 
                A list of paths having check point files and logs produced
                by TSN.
            cfg_file:
                Configuration file used to produce the runs.
        """
        self._dirs_ = dirs_

        # Creating model using config file
        cfg         = mmcv.Config.fromfile(cfg_file)
        self._model = build_recognizer(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

        # Create a df from all the logs
        self._df_frm_logs = self.load_log_files()

    def print_model(self, pth_file_path):
        """
        Prints TSN model loaded from saved file.
        """
        load_checkpoint(self._model, pth_file_path, strict=True)
        pdb.set_trace()


    def get_pca_projection(self, df, samples_per_run=10):
        """
        Computes PCA projection of samples.
        """
        # Loop over each run. Can be identified by unique values in "path"
        unique_paths = df.path.unique()
        df_dij = pd.DataFrame(columns=df.columns)
        
        for idx, cur_path in enumerate(unique_paths):
            cdf   = df[df["path"]==cur_path]
            # Getting last 10 points
            df_dij = df_dij.append(cdf.tail(samples_per_run), ignore_index=True)

        conv_flag_lst = []
        for i, row in df_dij.iterrows():
            print("Calculating projection")
            print(i)
            conv_flag = row["conv_flag"]
            conv_flag_lst = conv_flag_lst + [conv_flag]
            chkpt_file = row["path"] + "/epoch_" + str(row["epoch"]) + ".pth"
            load_checkpoint(self._model, chkpt_file, strict=True)
            ctensor  = torch.nn.utils.parameters_to_vector(self._model.parameters())

            cur_vec  = ctensor.detach().numpy()
            if i == 0:
                data_matrix = cur_vec
            else:
                data_matrix = np.vstack((data_matrix, cur_vec))

        pca = PCA(2)  # project to 2 dimensions
        projected = pca.fit_transform(data_matrix)
        return projected, conv_flag_lst, data_matrix


    def get_inter_dists(self, run1_dir, run2_dir, num_samples=20):
        """
        Calculates inter distances for "n" samples between two TSN runs.
        """
        # Loading run1 log file
        run1_log_file = list_files(run1_dir,[".log.json"])
        if len(run1_log_file) > 1:
            print("ERROR from get_inter_dists, Multiple log files")
            print("\t" + run1_dir)
            sys.exit()
        run1_df       = self.load_log_file(run1_log_file[0])

        # Loading run1 log file
        run2_log_file = list_files(run2_dir,[".log.json"])
        if len(run2_log_file) > 1:
            print("ERROR from get_inter_dists, Multiple log files")
            print("\t" + run2_dir)
            sys.exit()
        run2_df       = self.load_log_file(run2_log_file[0])

        if len(run1_df) != len(run2_df):
            print("ERROR: Different epochs")
            print("\t", run1_dir)
            print("\t", run2_dir)

        tot_num_epochs = len(run1_df)
        epoch_array = []
        dist_lst = []
        for i in range(1, num_samples):
            cur_epoch     = int(i*(tot_num_epochs/num_samples))
            epoch_array   = epoch_array + [cur_epoch]

            r1_chkpt_file = run1_dir + "epoch_" + str(cur_epoch) + ".pth"
            r2_chkpt_file = run2_dir + "epoch_" + str(cur_epoch) + ".pth"

            load_checkpoint(self._model, r1_chkpt_file, strict=True)
            r1tensor = torch.nn.utils.parameters_to_vector(self._model.parameters())

            load_checkpoint(self._model, r2_chkpt_file, strict=True)
            r2tensor = torch.nn.utils.parameters_to_vector(self._model.parameters())

            dist       = torch.dist(r1tensor, r2tensor,p=2)
            dist       = np.reshape(dist.detach().numpy(), (1,))

            dist_lst  = dist_lst + [dist]
        return dist_lst, epoch_array







    def calculate_dij(self, df,  samples_per_run=10, method="beginning_end"):
        """
        Calculates dij matrix with 10 samples take from each run.
        """ 
        # Loop over each run. Can be identified by unique values in "path"
        unique_paths = df.path.unique()
        df_dij = pd.DataFrame(columns=df.columns)
        
        labels = []
        for idx, cur_path in enumerate(unique_paths):
            cdf   = df[df["path"]==cur_path]
            cur_label = os.path.basename(cur_path)
            if method == "end":
                # Getting last 10 points at the end
                df_dij = df_dij.append(cdf.tail(samples_per_run), ignore_index=True)
                labels = labels + [cur_label+"_e"]*samples_per_run
            elif method == "beginning_end":
                # Getting first samples_per_run/2 and last samples_per_run/2
                df_dij = df_dij.append(cdf.head(int(samples_per_run/2)), ignore_index=True)
                labels = labels + [cur_label+"_b"]*int(samples_per_run/2)
                df_dij = df_dij.append(cdf.tail(int(samples_per_run/2)), ignore_index=True)
                labels = labels + [cur_label+"_e"]*int(samples_per_run/2)
            else:
                print("Method not supported ", method)


        # Loop over dij dataframe and calculate distances between them
        dij_npy = np.zeros((len(df_dij), len(df_dij)),dtype=float)
        dij_conv_npy = np.zeros((len(df_dij), len(df_dij)),dtype=float)
        for i, row in df_dij.iterrows():
            
            # Add 1 or 0 based on convergence of a point
            labels[i] = labels[i] + "_" + str(row["conv_flag"])

            i_conv_flag = bool(row["conv_flag"])
            chkpt_file = row["path"] + "/epoch_" + str(row["epoch"]) + ".pth"
            load_checkpoint(self._model, chkpt_file, strict=True)
            itensor = torch.nn.utils.parameters_to_vector(self._model.parameters())

            df_dj = df_dij.copy()
            df_dj = df_dj.drop(i)
            for j, row in df_dj.iterrows():
                if j > i:
                    j_conv_flag = bool(row["conv_flag"])
                    chkpt_file = row["path"] + "/epoch_" + str(row["epoch"]) + ".pth"
                    load_checkpoint(self._model, chkpt_file, strict=True)
                    jtensor = torch.nn.utils.parameters_to_vector(self._model.parameters())
                    
                    dist       = torch.dist(itensor, jtensor,p=2)
                    dist       = np.reshape(dist.detach().numpy(), (1,))
                    dij_npy[i,j] = dist
                    if i_conv_flag == True and j_conv_flag == True:
                        dij_conv_npy[i,j] = 1
                    if i_conv_flag == False and j_conv_flag == True:
                        dij_conv_npy[i,j] = 2
                    if i_conv_flag == True and j_conv_flag == False:
                        dij_conv_npy[i,j] = 2
                    if i_conv_flag == False and j_conv_flag == False:
                        dij_conv_npy[i,j] = 3
                print("(i,j) = ", i, j)
        return dij_npy, dij_conv_npy, labels


            


    def get_optimal_tensor(self, method="val_acc", plot_flag=False):
        """
        Returns optimal parameters(torch tensor) and performance.
        
        args:
            method(str):
                Method to use in determining optimal parameters.
                    1. val_acc = parameters producing best validation accuracy.

        Todo:
            1. Add more methods to determine optimal parameters.
        """

        # Picking the best data run and epoch
        if method=="val_acc":
            logsdf          = self._df_frm_logs.sort_values(by=["val_acc"], ascending=False)
            best            = dict(logsdf.iloc[0])
        elif method == "pareto_optimal":
            best            = self.get_pareto_optimal(plot_flag=plot_flag)
        else:
            print("The following method is not supported")
            print("\t", method)
            sys.exit(1)

        # Loading best checkpoint file
        best_chkpt_file = best["path"] + "/epoch_" + str(best["epoch"]) + ".pth"
        load_checkpoint(self._model, best_chkpt_file, strict=True)
        best_param_tensor = torch.nn.utils.parameters_to_vector(self._model.parameters())
        best_perf = np.array(best["val_acc"])

        # Return vector
        return best_param_tensor, best_perf


    def load_log_files(self):
        """
        Loads all log files as dataframe. The dataframe has following
        columns,
            1. log file path
            2. epoch
            3. validation accuracy
            4. Training loss
        """
        df = pd.DataFrame()
        
        # Loop over all the runs
        for crun_dir in self._dirs_:
            log_file = list_files(crun_dir,[".log.json"])
            
            # One log file per run should be present
            if len(log_file) != 1:
                print("ERROR: More than one log file found")
                print("\t", crun_dir)
                sys.exit(1)
                
            # Loading current log file as dataframe
            log_file = log_file[0]
            crun_df  = self.load_log_file(log_file)
            
            # Concatinating data frames from all the runs
            if df.empty:
                df = crun_df
            else:
                df = pd.concat([df, crun_df], ignore_index=True)
                
        # Return data frame having stats from all the runs
        return df
    

    def load_log_file(self, log_path):
        """
        Returns a data frame created using log file generated by TSN.

        args:
            log_path: Path to TSN log file.
        """
        val_acc = []
        tr_loss = []
        
        with open(log_path, "r") as f:
            json_lines = f.readlines()
            for line in json_lines:
                json_dict = json.loads(line)
                
                # Collect validation accuracy
                if json_dict["mode"] == "val":
                    val_acc = val_acc + [float(json_dict["top1 acc"])]
                    
                # Collect training loss
                if json_dict["mode"] == "train":
                    tr_loss = tr_loss + [float(json_dict["loss"])]
                    
            epochs        = list(range(1,len(tr_loss)+1))
            log_file_path = [os.path.dirname(log_path)]*len(tr_loss)
        df = pd.DataFrame(list(zip(log_file_path, epochs, val_acc, tr_loss)),
                          columns=["path","epoch","val_acc","tr_loss"])
        
        return df


    def mark_conv(self, df, th):
        """
        Marks each checkpoint as 0 or 1. 0 implying not converging and 1
        implying converging.

        args:
            th: Threshold used to determine a points convergance.

        """
        # Iterate over each point (row)
        conv_flags = [0]*len(df)
        for ridx, row in df.iterrows():
            row_last_epoch = df[df["path"] == row["path"]].iloc[-1]
            last_epoch_perf = row_last_epoch["val_acc"]
            if last_epoch_perf >= th:
                conv_flags[ridx] = 1

        df["conv_flag"] = conv_flags
        return df


    def log_dists_from_pivot(self, pivot_tensor):
        """
        Returns two lists, 
            1. List having distance from optimal tensor. 
            2. List having corresponding validation accuracy

        args:
            pivot_tensor: Tensor that acts as pivot. Form this all the distances 
                          are calculated.   
        """
        
        # Copy dataframe created from logs to local variable
        df = self._df_frm_logs.copy()
        
        # Iterate over each row and calculate distance from optimal
        dist_lst = []
        t = time.time()
        for ridx, row in df.iterrows():
            print("Percent completed = ", (ridx*100)/len(df))
            
            # Loading current checkpoint parameters as tensor
            chkpt_file = row["path"] + "/epoch_" + str(row["epoch"]) + ".pth"
            load_checkpoint(self._model, chkpt_file, strict=True)
            ctensor = torch.nn.utils.parameters_to_vector(self._model.parameters())

            # Calculating distance from pivot tensor
            dist       = torch.dist(pivot_tensor, ctensor,p=2)
            dist       = np.reshape(dist.detach().numpy(), (1,))
            dist_lst   = dist_lst + [dist[0]]
        # Adding distances in dataframe
        df["dist_from_pivot"] = dist_lst

        return df

    def get_pareto_optimal(self, plot_flag=False):
        """
        Returns a dictionary having pareto optimal point. The optimal point is
        found using Training loss and 1 - Validation accuracy
        """
        print("Calculating pareto optimal point")

        # Copying dataframe
        df = self._df_frm_logs.copy()

        # Getting x and y axis
        x = np.array(df["tr_loss"])
        y = 1 - np.array(df["val_acc"])

        # Loop through each point and mark it as pareto or not a pareto
        best_dist = math.inf
        pareto_dists = []
        pareto_flags = np.zeros(len(x),dtype=bool)
        for pidx in range(0,len(x)):
            x0 = x[pidx]
            y0 = y[pidx] 

            # points to left and below of (x0, y0)
            pts_left_x0y0     = x < x0 # left
            pts_below_x0y0    = y < y0 # below
            pts_better_x0y0   = np.multiply(pts_left_x0y0, pts_below_x0y0)
            num_better_points = sum(pts_better_x0y0)
            
            if num_better_points == 0:
                pareto_flags[pidx] = True
                dist = x0**2 + y0**2
                pareto_dists = pareto_dists + [dist]
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = pidx

        best_chkpt = dict(df.iloc[best_idx])
                
        # Plot pareto points
        if plot_flag:
            # Pareto points
            x_pareto = x[pareto_flags]
            y_pareto = y[pareto_flags]
            fig, ax = plt.subplots()
            ax.scatter(x,y)
            ax.scatter(x_pareto, y_pareto, c="r")
            ax.scatter(x[best_idx], y[best_idx], c="g")
            plt.show()
        return best_chkpt
import os
import sys
import pdb
import numpy as np
import pandas as pd
import tensorflow as tf
from .cnn3d_opt_frmwrks import OptFrmWrk
from ..print_tools import printof
from .cnn3d_architectures import CNN3DArchs
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


class CNN3D:
    """
    The following class provides an intuitive way to
        build custom neural networks using tensorflow 2
        for activity detection in trimmed videos.

    Todo:
        * After nested cross validation use n fold cross validation to get
            best of the best model.
    """

    def __init__(self, arch_params, training_params):
        """
        Initializes ParameterGrid with different parameters that can be
        varied in architecture and training as proved in the arguments.

        args:
            arch_params:  Parameters that define architecture.
            train_params: Training parameter dictionary.
        """
        # self._arch_params = arch_params
        # self._training_params = training_params
        all_params = {**arch_params, **training_params}
        self._var_params, self._con_params = self._get_var_con(all_params)

    def get_best_model(self, Xtr, ytr, f, method="nestedcv", ncv_split=(3, 3)):
        """
        Optimizes for best parameters and model using nested corss validation.

        Args:
            Xtr (nparray): An array having samples for training
            ytr (nparray): An array having labels corresponding to each sample in Xtr
            method (str) : A string having name of the parameter
                           parameter tuning method. Default is nested cross validation.
            ncv_split (tuple): Cross validation split for nestedcv,
                                (inner split, outer split). Default is (3,3)
        """
        # Getting optimal architecture and training parameters
        opt = OptFrmWrk(Xtr, ytr, self._var_params, self._con_params)
        if method == "nestedcv":

            ncv_best_params, perfs = opt.nested_cv(ncv_split, f)

            best_params = ncv_best_params[np.argmax(perfs)]
            Xsplt_tr, Xsplt_ts, ysplt_tr, ysplt_ts = train_test_split(
                Xtr, ytr, test_size=0.20, random_state=42
            )
            perf_lst, model_lst = opt.get_median_perf(
                Xsplt_tr, ysplt_tr, Xsplt_ts, ysplt_ts, best_params
            )

            median_perf = np.median(perf_lst)
            median_model = model_lst[perf_lst.index(median_perf)]
            printof("\n\nAfter Nested CV(Training best of best", f)
            printof("\tParams: " + str(best_params), f)
            printof("\tMedian(best_of_best): " + str(median_perf), f)
            printof("\t\tList(best_of_best): " + str(perf_lst), f)
        else:
            print("Parameter tuning not supported")
            sys.exit()

        return best_params, median_model

    def eval_combinations(self, Xtr, ytr, Xvl, yvl):
        """
        Evaluates all combinations.

        Args:
            Xtr (nparray): An array having samples for training
            ytr (nparray): An array having labels corresponding to each sample in Xtr
            Xvl (nparray): An array having samples for validation
            yvl (nparray): An array having labels corresponding to each sample in Xvl
        """
        print("\n--- Evaluating best parameters ---")
        opt = OptFrmWrk(Xtr, ytr, self._var_params, self._con_params)
        param_grid = ParameterGrid(self._var_params)
        
        best_perf = -np.inf
        all_dict = dict.fromkeys(self._var_params,[])
        all_dict["perf"] = []
        for pidx, cparams in enumerate(param_grid):
            print("\t",cparams)
            for k in cparams:
                all_dict[k] = all_dict[k] + [cparams[k]]
            
            # Build model
            num_runs = 1
            all_cparams = {**cparams, **self._con_params}
            perf_lst, model_lst = opt.train_n_times(
                Xtr, ytr, Xvl, yvl, all_cparams, n=num_runs
            )
            max_perf = np.max(perf_lst)

            
            all_dict["perf"] = all_dict["perf"] + [max_perf]
            
            if max_perf > best_perf:
                best_perf =  max_perf
                best_model = model_lst[perf_lst.index(best_perf)]
                best_params = cparams
                best_dict = {"params":best_params, "perf":best_perf, "model":best_model}

        return all_dict, best_dict

    def _get_var_con(self, params):
        """
        Divides parameters into constants and variables.
        Args:
            params (dict): All the parameters in an array
        """
        var_dict = {}
        con_dict = {}
        for key in params:
            if len(params[key]) > 1:
                var_dict[key] = params[key]
            else:
                con_dict[key] = params[key][0]

        return var_dict, con_dict

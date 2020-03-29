# coding: utf-8

import sys
import pdb
import time
import datetime
import numpy as np
import pandas as pd
from .cnn3d_architectures import CNN3DArchs
from ..print_tools import printof
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow as tf


class OptFrmWrk:
    """
    Optimization frameworks to explore and find best models.

    Todo:
        * Use singel cv inside nested cv
    """

    def __init__(self, Xtr, ytr, vars_, cons_):
        """
        Args:
            X(np_array): A numpy array having training samples
            y (np._array): A numpy array having labels
            vars_ (dict): Dictionary having parameters that can be varied
            cons_ (dict): Dictionary having parameters that are constants
        Quote:
            “Constants and variables.” AND “There’s always a lighthouse.
            There’s always a man, there’s always a city…”
            --- Bioshock: infinite
        """
        self._Xtr = Xtr
        self._ytr = ytr
        self._cons = cons_
        self._vars = vars_

    def train_n_times(self, Xtr, ytr, Xts, yts, params, n=3, tfboard=True):
        """
        Trains an achitecutre with specific training data "n" times
        and returns median performance and model.

        Args:
            Xtr(np_array) : A numpy array having training samples
            ytr(np_array) : A numpy array having training labels
            Xts(np_array) : A numpy array having testing samples
            yts(np_array) : A numpy array having testing labels
            params(dict): A dictionary having parameters to build architecture.
            n(int)      : Number(odd) times training is performed. By default
                          it is 5.
        """
        _tfboard_dir = params["tfboard_dir"]
        log_dir = _tfboard_dir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = log_dir
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

        perf_lst = []
        model_lst = []
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5
        )
        for idx in range(0, n):
            model = CNN3DArchs(params, Xtr, ytr).build_model()
            epochs_ = params["epochs"]
            batch_size_ = params["batch_size"]
            model.fit(
                Xtr,
                ytr,
                epochs=epochs_,
                batch_size=batch_size_,
                validation_data=[Xts, yts],
                verbose=1
            )
            in_loss, in_perf = model.evaluate(Xts, yts, verbose=0)
            # in_pred = 1 * (model.predict(Xts.astype("float32")) > 0.5)
            # conf_mat = confusion_matrix(yts, in_pred)
            perf_lst.append(in_perf)
            model_lst.append(model)
            # Clearing session and waiting for 5 seconds to be sure
            tf.keras.backend.clear_session()
            del model

        return perf_lst, model_lst

    def nested_cv(self, split, f):
        """
        Optimizes for best parameters and model using nested cv.

        Args:
            params (dict): Dictionary of parameters to optimize
            split (tuple): A tuple having cross validation split parts.
                (inner split, outer split)
        """
        param_grid = ParameterGrid(self._vars)

        in_cv = StratifiedKFold(split[0])
        out_cv = StratifiedKFold(split[1])

        # Outer cross validation loop
        best_perfs = []
        best_params_lst = []
        for out_tr_idx, out_tst_idx in out_cv.split(self._Xtr, self._ytr):
            printof("Outer CV", f)

            # Parameter loop
            param_best_perf = -np.inf
            for pidx, cparams in enumerate(param_grid):
                printof("\tParameters loop " + str(cparams), f)

                # Inner cross validation loop
                in_perfs = []
                for in_tr_idx, in_tst_idx in in_cv.split(
                    self._Xtr[out_tr_idx], self._ytr[out_tr_idx]
                ):
                    all_cparams = {**cparams, **self._cons}
                    perf_lst, _ = self.get_median_perf(
                        self._Xtr[in_tr_idx],
                        self._ytr[in_tr_idx],
                        self._Xtr[in_tst_idx],
                        self._ytr[in_tst_idx],
                        all_cparams,
                    )

                    median_perf = np.median(perf_lst)
                    in_perfs.append(median_perf)
                    printof("\t\tMedain(in): " + str(median_perf), f)
                    printof("\t\t\tList(in): " + str(perf_lst), f)

                # Mean inner performance
                in_mean_perf = np.mean(in_perfs)
                printof("\t\tMean(in): " + str(in_mean_perf), f)
                if in_mean_perf > param_best_perf:
                    param_best_perf = in_mean_perf
                    best_params = cparams

            printof("\tInner best parameters " + str(best_params), f)
            printof("\tMean Best performance " + str(param_best_perf), f)

            # Performance of best parameters on outer split
            all_cparams = {**best_params, **self._cons}
            perf_lst_out, _ = self.get_median_perf(
                self._Xtr[out_tr_idx],
                self._ytr[out_tr_idx],
                self._Xtr[out_tst_idx],
                self._ytr[out_tst_idx],
                all_cparams,
            )
            median_perf_out = np.median(perf_lst_out)
            printof("Best parameters " + str(best_params), f)
            printof("Median(out): " + str(median_perf_out), f)
            printof("\t" + str(perf_lst_out) + "\n", f)

            # Storing best parameters for outer loop
            best_params_lst.append({**self._cons, **best_params})
            best_perfs.append(median_perf_out)
        return best_params_lst, best_perfs

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import pdb
import cv2
import numpy as np
import pprint

from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import keras
from keras.models import Model
from keras.models import load_model
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import binary_crossentropy


class SampleDNN:
    """
    SampleDNN simply returns a keras model. It does not
    replace the functional api.
    """

    def __init__(
        self,
        input_size,
        num_first_filters,
        num_conv_nets,
        kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
    ):
        """
        Class having methods and techniques to explore DNN
        classifiers using keras library. 
        
        The model is saved in <object name>.model_

        Args:
            input_size (tuple)  : (num_frames, num_rows, num_cols, num_channels)
                1. writing no writing uv np arrays = (89,  100, 100, 2)
                2. talking no talking uv np arrays = (299, 100, 100, 2)
                3. writing no writing gray np arrays = (90,  100, 100, 1)
                4. talking no talking gray np arrays = (300,  100, 100, 1)

            num_first_filters   : Number of first layer filters

            num_conv_nets (int) : Number of ConvNets

            kernel_size (tuple) : (num_frames, num_rows, num_cols)
                                  Default = (3,3,3)
        """
        self.input_size_ = input_size
        self.num_first_filters_ = num_first_filters
        self.num_conv_nets_ = num_conv_nets
        self.kernel_size_ = kernel_size
        self.pool_size_ = pool_size
        self.model_ = self.build()

    def __repr__(self):
        """
        Provides internal information for developers
        """
        info = """
            The SampleDNN class provides examples for you to
            build your own class.
        """

    def __str__(self):
        """
        Prints model summary after the build method.
        
        Example:
            print(name_of_the_object)
        """
        info = (
            "Architecture:\n"
            + "    Input size = "
            + str(self.input_size_)
            + "    Number of first filters = "
            + str(self.num_first_filters_)
            + "    Number of ConvNets = "
            + str(self.num_conv_nets_)
        )
        return info

    def build(self):
        """
        Builds keras model using parameters provided.
        """
        input_layer = Input(self.input_size_)
        ## convolutional layers
        conv_layer = Conv3D(
            filters=self.num_first_filters_,
            kernel_size=self.kernel_size_,
            activation="relu",
            data_format="channels_last",
        )(input_layer)
        pool_layer = MaxPool3D(pool_size=self.pool_size_, data_format="channels_last")(
            conv_layer
        )

        for i in range(self.num_conv_nets_ - 1):
            conv_layer = Conv3D(
                filters=self.num_first_filters_,
                kernel_size=self.kernel_size_,
                activation="relu",
                data_format="channels_last",
            )(pool_layer)
            pool_layer = MaxPool3D(
                pool_size=self.pool_size_, data_format="channels_last"
            )(conv_layer)

        ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
        pool_layer = BatchNormalization()(pool_layer)
        flatten_layer = Flatten()(pool_layer)

        ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
        ## add dropouts to avoid overfitting / perform regularization

        dense_layer1 = Dense(units=128, activation="relu")(flatten_layer)
        dense_layer1 = Dropout(0.4)(dense_layer1)
        # dense_layer2 = Dense(units=32, activation='relu')(dense_layer1)
        # dense_layer2 = Dropout(0.4)(dense_layer2)

        output_layer = Dense(units=1, activation="sigmoid")(dense_layer1)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss=binary_crossentropy, optimizer="sgd", metrics=["acc"])

        return self.model


def nested_cv(X, y, inner_cv, outer_cv, DNN, param_grid, tr_batch_size=1, tr_epochs=10):
    """
    This function performs nested cross validation and returns
    and saves best model.
    Args:
        X (np array)    : Training numpy array.
                           X[0] gives first sample
                           X[r] gives range of values in r.
        y (np array)    : A numpy vector string the labels corresponding to X.
        inner_cv        : inner loop cross validation split
        outer_cv        : outer loop cross validation split
        DNN             : Keras model to train
        param_grid      : Hyper parameters values to optimize
        tr_batch_size   : Number of X-Samples per iteration
        tr_epochs       : Number of times to train using the same sample
                            size as a training data
    """
    train_history = []
    all_best_params = []
    all_test_loss = []
    all_test_metric = []
    # for each split of the data in the outer cross-validation
    # (split method returns indices of training and test parts)
    #
    out_split_idx = 1
    num_out_splits = outer_cv.get_n_splits(X, y)
    for training_samples, test_samples in outer_cv.split(X, y):
        print("Outer split: " + str(out_split_idx) + "/" + str(num_out_splits))
        out_split_idx += 1
        # find best parameter using inner cross-validation
        best_params = {}
        best_metric = -np.inf
        # iterate over parameters

        for par_idx, parameters in enumerate(param_grid):
            print("\tParameter : " + str(par_idx + 1) + "/" + str(len(param_grid)))
            print("\t", parameters)
            # accumulate score over inner splits
            cv_loss = []
            cv_metric = []
            # iterate over inner cross-validation
            in_split_idx = 1
            num_in_splits = inner_cv.get_n_splits(
                X[training_samples], y[training_samples]
            )
            in_start_time = time.time()
            for inner_train, inner_test in inner_cv.split(
                X[training_samples], y[training_samples]
            ):
                # build classifier given parameters and training data
                keras.backend.clear_session()
                dnn_inst = DNN(**parameters)
                keras_model = dnn_inst.model_
                in_start_time = time.time()
                keras_model.fit(
                    X[inner_train],
                    y[inner_train],
                    epochs=tr_epochs,
                    validation_split=0.2,
                    batch_size=tr_batch_size,
                    verbose=0,
                )

                in_end_time = time.time()
                in_time_taken = in_end_time - in_start_time
                print(
                    "\t\tInner split: "
                    + str(in_split_idx)
                    + "/"
                    + str(num_in_splits)
                    + ", "
                    + str(round(in_end_time - in_start_time))
                    + " sec"
                )
                in_split_idx += 1

                # evaluate on inner test set
                loss, metric = keras_model.evaluate(
                    X[inner_test], y[inner_test], verbose=0
                )
                cv_loss.append(loss)
                cv_metric.append(metric)

            # compute mean score over inner folds
            # for a single combination of parameters.
            mean_loss = np.mean(cv_loss)
            mean_metric = np.mean(cv_metric)
            print("\t\tMean loss :" + str(mean_loss))
            print("\t\tMean Metric:" + str(mean_metric))
            if mean_metric > best_metric:
                # if better than so far, remember parameters
                best_metric = mean_metric
                best_loss = mean_loss
                best_params = parameters

            # Save training results
            param_history = {
                "cv_loss": cv_loss,
                "cv_metric": cv_metric,
                "cv_params": parameters,
                "mean_loss": mean_loss,
                "mean_metric": mean_metric,
                "train_best_mean_loss": best_loss,
                "train_best_mean_metric": best_metric,
            }
            # print(param_history)
            train_history.append(param_history)

        # Build classifier on best parameters using outer training set
        # This is done over all parameters evaluated through a single
        # outer fold and all inner folds.
        # Fitting with best parameters and testing on
        # outer fold
        print("\tTraining on best prameters")
        best_dnn = SampleDNN(**best_params)
        keras_model = best_dnn.model_
        keras_model.fit(
            X[training_samples],
            y[training_samples],
            epochs=all_epochs,
            batch_size=all_batches,
            verbose=0,
        )
        test_loss, test_metric = keras_model.evaluate(
            X[test_samples], y[test_samples], verbose=0
        )
        print("\tLoss:" + str(test_loss))
        print("\tMetric:" + str(test_metric))
        # Outer loop test results:
        all_best_params.append(best_params)
        all_test_loss.append(test_loss)
        all_test_metric.append(test_metric)

    train_results = {
        "train_history": train_history,
        "test_metric": all_test_metric,
        "best_params": all_best_params,
        "test_loss": all_test_loss,
    }
    return train_results


X_train_ori = np.load("./nparrays/wnw/train_flow_X_30.npy")
y_train_ori = np.load("./nparrays/wnw/train_flow_y_30.npy")

# `in_size` for
#     1. writing no writing uv np arrays = (89,  100, 100, 2)
#     2. talking no talking uv np arrays = (299, 100, 100, 2)
#     3. writing no writing gray np arrays = (90,  100, 100, 1)
#     4. talking no talking gray np arrays = (300,  100, 100, 1)
in_size = [(89, 50, 50, 2)]
num_first_fil = [2, 4]
num_convnets = [2, 3]
param_grid = {
    "input_size": in_size,
    "num_first_filters": num_first_fil,
    "num_conv_nets": num_convnets,
}

all_batches = 10
all_epochs = 10

train_results = nested_cv(
    X_train_ori,
    y_train_ori,
    StratifiedKFold(3),
    StratifiedKFold(3),
    SampleDNN,
    ParameterGrid(param_grid),
    tr_batch_size=all_batches,
    tr_epochs=all_epochs,
)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(train_results)
print("\n\n")
scores = np.array(train_results["test_metric"])
print("Cross-validation scores: {}".format(scores))
stats_list = (scores.min(), scores.max(), scores.mean())
print("Min value = {:0.2f}, Max value = {:0.2f}, Mean = {:0.2f}".format(*stats_list))


# Split 80% for training and 20% for testing
X_train, X_val, y_train, y_val = train_test_split(
    X_train_ori, y_train_ori, test_size=0.2
)
best_of_all_metric = -np.inf
tst = train_results["best_params"]
for i in range(len(tst)):

    # Train on 80%
    keras.backend.clear_session()
    best_dnn = SampleDNN(**tst[i])
    keras_model = best_dnn.model_
    keras_model.fit(
        X_train, y_train, epochs=all_epochs, batch_size=all_batches, verbose=0
    )

    loss, metric = keras_model.evaluate(X_val, y_val)

    # Determine the best of the best
    if metric > best_of_all_metric:
        best_of_all_params = tst[i]
        best_of_all_loss = loss
        best_of_all_metric = metric

print("\n============== BEST OF ALL =============")
print(best_of_all_params)
print("Best of all loss: " + str(best_of_all_loss))
print("Best of all metric: " + str(best_of_all_metric))
print("========================================\n")


print("Training on entire training set with best of all parameters")
keras.backend.clear_session()
best_dnn = SampleDNN(**best_of_all_params)
print(best_dnn)

keras_model = best_dnn.model_
keras_model.fit(
    X_train_ori, y_train_ori, epochs=all_epochs, batch_size=all_batches, verbose=0
)
keras_model.save("best_model_flow.h5")
# Delete the existing model.
del keras_model


# TESTING PHASE
X_test = np.load("./nparrays/wnw/test_flow_X_30.npy")
y_test = np.load("./nparrays/wnw/test_flow_y_30.npy")
model = load_model("best_model_flow.h5")
y_pred = 1 * (model.predict(X_test) > 0.5)
y_pred = y_pred.flatten()
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("\n Accuracy on test set: " + str(accuracy))
print(conf_matrix)

# Testing on training to check learning
y_pred = 1 * (model.predict(X_train) > 0.5)
y_pred = y_pred.flatten()
conf_matrix = confusion_matrix(y_train, y_pred)
accuracy = accuracy_score(y_train, y_pred)
print("\nAccuracy on train set: " + str(accuracy))
print(conf_matrix)

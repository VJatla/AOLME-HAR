import math
import sys
import pdb
import tensorflow as tf
import tensorflow.keras.layers as tfkr_layers


class CNN3DArchs:
    """
    Contains methods to build 3D CNN architectures. The architecture is a tf.keras
    model and supports funtional api of tf.keras.
    """

    def __init__(self, params, X, y):
        """
        Returns a tf.keras model
        """
        self._X = X
        self._y = y
        self._params = params

    def build_model(self):
        """
        Builds a tf.keras model based the parameters.

        Args:
            params (dict):
                Parameters to use for building model.
        """
        arch_name = self._params["arch_name"]
        if arch_name == "custom":
            model = self._build_custom_model()
        elif arch_name == "custom_c3d":
            model = self._build_custom_c3d_model()
        else:
            print("Architecture not supported ", arch_name)
            sys.exit()
        return model

    def _build_custom_c3d_model(self):
        """
        An architecture based on C3D proposed in
        "Learning Spatiotemporal Features with 3D Convolutional Networks"
        """
        # Custamizable parameters
        _num_kernels = self._params["num_kernels"]
        _num_fcn_units = self._params["num_fcn_units"]

        # Input Layer
        sample_shape = self._X.shape[1:]
        # mirrored_strategy = tf.distribute.MirroredStrategy()
        # mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
        with tf.device("/device:GPU:1"):
            #with mirrored_strategy.scope():
            input_layer = tfkr_layers.Input(sample_shape)
            # norm1, conv1, pool1
            norm1 = tfkr_layers.BatchNormalization(name="norm1")(input_layer)
            conv1 = tfkr_layers.Conv3D(
                filters=_num_kernels,
                kernel_size=3,
                activation="relu",
                data_format="channels_last",
                padding="same",
                name="conv1",
            )(norm1)
            pool1 = tfkr_layers.MaxPooling3D(
                pool_size=(3, 2, 2), data_format="channels_last", name="pool1"
            )(conv1)


            # norm2a, conv2a, norm2b, conv2b, pool2
            norm2a = tfkr_layers.BatchNormalization(name="norm2a")(pool1)
            conv2a = tfkr_layers.Conv3D(
                filters=_num_kernels,
                kernel_size=3,
                activation="relu",
                padding="same",
                data_format="channels_last",
            )(norm2a)
            norm2b = tfkr_layers.BatchNormalization(name="norm2b")(conv2a)
            conv2b = tfkr_layers.Conv3D(
                filters=_num_kernels,
                kernel_size=3,
                activation="relu",
                padding="same",
                data_format="channels_last",
                name="conv2b",
            )(norm2b)
            pool2 = tfkr_layers.MaxPooling3D(
                pool_size=(3, 2, 2), data_format="channels_last", name="pool2"
            )(conv2b)

            # norm3a, conv3a, norm3b, conv3b, pool3
            norm3a = tfkr_layers.BatchNormalization(name="norm3a")(pool2)
            conv3a = tfkr_layers.Conv3D(
                filters=_num_kernels,
                kernel_size=3,
                activation="relu",
                padding="same",
                data_format="channels_last",
                name="conv3a",
            )(norm3a)
            norm3b = tfkr_layers.BatchNormalization(name="norm3b")(conv3a)
            conv3b = tfkr_layers.Conv3D(
                filters=_num_kernels,
                kernel_size=3,
                activation="relu",
                padding="same",
                data_format="channels_last",
                name="conv3b",
            )(norm3b)
            pool3 = tfkr_layers.MaxPooling3D(
                pool_size=(5, 1, 1), data_format="channels_last", name="pool3"
            )(conv3b)

            # norm4a, conv4a, norm4b, conv4b, pool4
            norm4a = tfkr_layers.BatchNormalization(name="norm4a")(pool3)
            conv4a = tfkr_layers.Conv3D(filters = _num_kernels,
                                        kernel_size = 3,
                                        activation = "relu",
                                        data_format = "channels_last",
                                        padding="same",
                                        name="conv4a")(norm4a)
            norm4b = tfkr_layers.BatchNormalization(name="norm4b")(conv4a)
            conv4b = tfkr_layers.Conv3D(filters = _num_kernels,
                                        kernel_size = 3,
                                        activation = "relu",
                                        data_format = "channels_last",
                                        padding="same",
                                        name="conv4b")(norm4b)
            pool4 = tfkr_layers.MaxPooling3D(pool_size=(2,1,1),
                                             data_format="channels_last",
                                             name="pool4")(conv4b)

            # flat_layer
            flat_layer = tfkr_layers.Flatten()(pool4)
            
            # norm4a, fcn1, norm4b, fcn2
            norm_fcn1 = tfkr_layers.BatchNormalization(name="norm_fcn1")(flat_layer)
            fcn1 = tfkr_layers.Dense(
                units=_num_fcn_units,
                activation="relu",
                kernel_initializer="glorot_normal",
                name="fcn1",
            )(norm_fcn1)
            fcn1_drop_out = tfkr_layers.Dropout(0.5)(fcn1)
            norm_fcn2 = tfkr_layers.BatchNormalization(name="norm_fcn2")(fcn1_drop_out)
            fcn2 = tfkr_layers.Dense(
                units=_num_fcn_units,
                activation="relu",
                kernel_initializer="glorot_normal",
                name="fcn2",
            )(norm_fcn2)
            fcn2_dropout = tfkr_layers.Dropout(0.5)(fcn1)

            # output layer, sigmoid for binary classificaiton
            output_layer = tfkr_layers.Dense(units=1, activation="sigmoid")(
                fcn2_dropout
            )

            # Return model

            model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
            model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["acc"])
        return model

    def _build_custom_model(self):
        """
        Builds a flat cnn3d model using tensorflow 2. It has same
        number of convolutional kernels throughout.
        Args:
            params (dict): Dictionary having parameters for architecture.
                1. num_conv_layers
                2. num_kernels
        """
        # Extracting architecture parameters from dictionary
        kernels_per_layer_ = self._params["kernels_per_layer"]
        kernel_size_ = self._params["kernel_size"]
        activation_ = self._params["activation"]
        data_format_ = self._params["data_format"]
        pool_size_ = self._params["pool_size"]
        pool_type_ = self._params["pool_type"]
        batch_norm_ = self._params["batch_norm"]
        drop_out_ = self._params["drop_out"]
        num_dense_layers_ = self._params["num_dense_layers"]
        final_activation_ = self._params["final_activation"]
        loss_ = self._params["loss"]
        optimizer_ = self._params["optimizer"]
        dense_units_ = self._params["dense_units"]
        metric_ = self._params["metric"]
        num_gpus_ = self._params["num_gpus"]

        # Input Layer
        sample_shape = self._X.shape[1:]
        input_layer = tfkr_layers.Input(sample_shape)

        # Batch Normalization
        if batch_norm_ == "before_conv3d":
            norm_input_layer = tfkr_layers.BatchNormalization()(input_layer)
            # First convoluton and pooling layers
            conv_layer = tfkr_layers.Conv3D(
                filters=kernels_per_layer_[0],
                kernel_size=kernel_size_,
                activation=activation_,
                data_format=data_format_,
                kernel_initializer="glorot_normal",
            )(norm_input_layer)
        else:
            conv_layer = tfkr_layers.Conv3D(
                filters=kernels_per_layer_[0],
                kernel_size=kernel_size_,
                activation=activation_,
                data_format=data_format_,
                kernel_initializer="glorot_normal",
            )(input_layer)

        if pool_type_ == "Max":
            pool_layer = tfkr_layers.MaxPool3D(
                pool_size=pool_size_, data_format=data_format_
            )(conv_layer)
        elif pool_type_ == "Avg":
            pool_layer = tfkr_layers.AveragePooling3D(
                pool_size=pool_size_, data_format=data_format_
            )(conv_layer)
        else:
            print("Pooling layer not supported ", pool_type_)
            sys.exit()

        # Remaining convolution and pooling layers
        for layer_idx in range(1, len(kernels_per_layer_)):
            conv_layer = tfkr_layers.Conv3D(
                filters=kernels_per_layer_[layer_idx],
                kernel_size=kernel_size_,
                activation=activation_,
                data_format=data_format_,
                kernel_initializer="glorot_normal",
            )(pool_layer)

            pool_layer = tfkr_layers.MaxPool3D(
                pool_size=pool_size_, data_format=data_format_
            )(conv_layer)

        # Batch Normalization
        if batch_norm_ == "after_conv3d":
            pool_layer = tfkr_layers.BatchNormalization()(pool_layer)

        # Flatten
        flat_layer = tfkr_layers.Flatten()(pool_layer)

        # dense/dropout layers
        dense_layer = tfkr_layers.Dense(
            units=dense_units_,
            activation=activation_,
            kernel_initializer="glorot_normal",
        )(flat_layer)
        if drop_out_:
            dense_layer = tfkr_layers.Dropout(0.4)(dense_layer)

        for layer_idx in range(1, num_dense_layers_):
            dense_layer = tfkr_layers.Dense(
                units=dense_units_,
                activation=activation_,
                kernel_initializer="glorot_normal",
            )(dense_layer)
            if drop_out_:
                dense_layer = tfkr_layers.Dropout(0.4)(dense_layer)

        # output layer, sigmoid for binary classificaiton
        output_layer = tfkr_layers.Dense(units=1, activation=final_activation_)(
            dense_layer
        )

        # Return model
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        if num_gpus_ > 1:
            model = tf.keras.utils.multi_gpu_model(model, gpus=num_gpus_)
        model.compile(loss=loss_, optimizer=optimizer_, metrics=[metric_])

        return model
